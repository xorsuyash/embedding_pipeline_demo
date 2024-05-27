import pandas as pd
from ragatouille import RAGTrainer
import os
import torch
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher
from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker
from colbert.infra.config import ColBERTConfig
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints
import torch.distributed as dist


class Model:
    _instance = None

    def __new__(cls, lr=1e-05, model_checkpoint="colbert-ir/colbertv2.0"):
        if cls._instance is None:
            cls._instance = super(Model, cls).__new__(cls)
            cls._instance.initialize_model(lr, model_checkpoint)
        return cls._instance

    def initialize_model(self, lr, model_checkpoint):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = ColBERTConfig(
            bsize=1,
            lr=lr,
            warmup=3000,
            doc_maxlen=180,
            dim=128,
            attend_to_mask_tokens=False,
            nway=2,
            accumsteps=1,
            similarity="cosine",
            use_ib_negatives=True,
            checkpoint=model_checkpoint,
        )

        assert self.config.bsize % self.config.nranks == 0, (
            self.config.bsize,
            self.config.nranks,
        )
        self.config.bsize = self.config.bsize // self.config.nranks

        # model
        if not self.config.reranker:
            self.colbert = ColBERT(name=self.config.checkpoint, colbert_config=self.config)
        else:
            self.colbert = ElectraReranker.from_pretrained(self.config.checkpoint)

        self.colbert = self.colbert.to(self.device)
        self.colbert.train()

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.colbert.parameters()),
            lr=self.config.lr,
            eps=1e-8,
        )
        self.optimizer.zero_grad()

        self.scheduler = None
        if self.config.warmup is not None:
            print(
                f"#> LR will use {self.config.warmup} warmup steps and linear decay over {self.config.maxsteps} steps."
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.warmup, gamma=0.1
            )

    def get_model(self):
        return self.colbert

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.scheduler
    
    def get_config(self):
        return self.config


def get_reader(collection, config, triples, queries):
    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(
                config,
                triples,
                queries,
                collection,
                (0 if config.rank == -1 else config.rank),
                config.nranks,
            )
        else:
            reader = LazyBatcher(
                config,
                triples,
                queries,
                collection,
                (0 if config.rank == -1 else config.rank),
                config.nranks,
            )
        return reader
    else:
        raise NotImplementedError()

def train(triples,
          query_path,
          collection_path,
          model_checkpoint,
          lr=1e-05,
          num_epochs=1,):
    
    DEVICE="cuda" if torch.cuda.is_available() else "cpu"

    collection_path=os.path.join("mined_data",collection_path)
    triples=os.path.join("mined_data",triples)
    query_path=os.path.join("mined_data",query_path)

    model_instance=Model(lr=lr,model_checkpoint=model_checkpoint)

    colbert_model=model_instance.get_model()
    optimizer=model_instance.get_optimizer()
    scheduler=model_instance.get_scheduler()
    config=model_instance.get_config()

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)
    
    train_loss = None
    train_loss_mu = 0.999
    start_batch_idx = 0

    loss_df = pd.DataFrame(columns=["Epoch", "Step", "Loss"])
    loss_history = []
    checkpoint_paths=[]

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")

        # Reinitialize or reset the reader here if necessary
        reader = get_reader(collection_path, config, triples, query_path)

        for batch_idx, BatchSteps in zip(range(start_batch_idx, 256), reader):
            # if (warmup_bert is not None) and warmup_bert <= batch_idx:
            #     set_bert_grad(colbert, True)
            #     warmup_bert = None
            this_batch_loss = 0.0

            for batch in BatchSteps:
                with amp.context():
                    try:
                        queries, passages, target_scores = batch
                        encoding = [queries, passages]
                    except:
                        encoding, target_scores = batch
                        encoding = [encoding.to(DEVICE)]

                    scores = colbert_model(*encoding)

                    if config.use_ib_negatives:
                        scores, ib_loss = scores

                    scores = scores.view(-1, config.nway)

                    if len(target_scores) and not config.ignore_scores:
                        target_scores = (
                            torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                        )
                        target_scores = target_scores * config.distillation_alpha
                        target_scores = torch.nn.functional.log_softmax(
                            target_scores, dim=-1
                        )

                        log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                        loss = torch.nn.KLDivLoss(
                            reduction="batchmean", log_target=True
                        )(log_scores, target_scores)
                    else:
                        loss = nn.CrossEntropyLoss()(scores, labels[: scores.size(0)])

                    if config.use_ib_negatives:
                        if config.rank < 1:
                            print(
                                "EPOCH ",
                                epoch,
                                " \t\t\t\t",
                                loss.item(),
                                ib_loss.item(),
                            )

                        loss += ib_loss

                    loss = loss / config.accumsteps

                if config.rank < 1:
                    print_progress(scores)

                amp.backward(loss)
                this_batch_loss += loss.item()

                if batch_idx % 500 == 0:
                    formatted_loss = "{:.6e}".format(
                        this_batch_loss
                    )  # Adjust the precision (e.g., 6) as needed
                    loss_history.append((epoch + 1, batch_idx + 1, formatted_loss))
                    loss_df = pd.DataFrame(
                        loss_history, columns=["Epoch", "Step", "Loss"]
                    )
                    loss_path="loss_history.csv"
                    loss_df.to_csv(loss_path, index=False)

            train_loss = this_batch_loss if train_loss is None else train_loss
            train_loss = (
                train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss
            )

            amp.step(colbert_model, optimizer, scheduler)

        if config.rank < 1:
            print_message(batch_idx, train_loss)
            epoch_save_path = f"./model_checkpoint/epoch_{epoch}/"
            os.makedirs(epoch_save_path, exist_ok=True)
            checkpoint_filename = f"checkpoint_batch_{batch_idx+1}.pt"
            full_checkpoint_path = os.path.join(epoch_save_path, checkpoint_filename)
            manage_checkpoints(
                config,
                colbert_model,
                optimizer,
                batch_idx + 1,
                savepath=full_checkpoint_path,
                consumed_all_triples=True,
            )
            config.checkpoint = full_checkpoint_path + "/colbert/"
            checkpoint_paths.append(full_checkpoint_path)

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(
            config,
            colbert_model,
            optimizer,
            batch_idx + 1,
            savepath=None,
            consumed_all_triples=True,
        )
    
    return loss_path,checkpoint_paths
            
