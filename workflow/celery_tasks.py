from celery import shared_task
import uuid 
import logging 
import os 
import pandas as pd 
from .models import DatasetProcessingTask,NegativeMiningResult,RetrievalModel
import multiprocessing
multiprocessing.set_start_method('spawn')

from ragatouille import RAGTrainer
from .retriever_model import train 
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

logger=logging.getLogger(__name__)

HUGGINFACE_API_KEY=os.getenv("HUGGING_FACE_API_KEY")


@shared_task(bind=True)
def process_dataset(self, task_pk):
    try:
        task = DatasetProcessingTask.objects.get(pk=task_pk)
        task.status='PROCESSING'
        task.save()
        df = pd.read_csv(task.dataset_path)
        print(f"Processing dataset from {task.dataset_path}")
        print(df.head(2))

        # Update the task state with the results of the processing
        task.status='COMPLETE'
        task.save()
    except Exception as e:
        logger.error(f"Error processing dataset at {task.dataset_path}:{str(e)}")

        task.status='FAILED'
        task.save()
        # Raise an exception to retry the task
        raise self.retry(exc=e, countdown=60, max_retries=1)

@shared_task(bind=True)
def mine_negatives(self,task_pk):

    try:
        task = DatasetProcessingTask.objects.get(pk=task_pk)
        task_id=task.task_id
        if task.status !='COMPLETE':
             raise Exception(f'Task {task_id} is not complete yet')

        mine_task=NegativeMiningResult.objects.get(task_id=task_id)

        
        dataset=pd.read_csv(task.dataset_path)

        output_data_path='./mined_data/'

        train_pairs=[(r["question"],r["answer"]) for _, r in dataset[["question","answer"]].iterrows()]

        content=dataset["oracle_context"]

        mine_task.status='PROCESSING'
        mine_task.save()
        
        trainer=RAGTrainer(model_name=mine_task.model_name,pretrained_model_name=mine_task.model_checkpoint)

        trainer.prepare_training_data(raw_data=train_pairs,
                                      data_out_path=output_data_path,
                                      all_documents=content.to_list(),
                                      num_new_negatives=10,
                                      mine_hard_negatives=True)
        
        files=os.listdir(output_data_path)

        for file in files:
             if file.startswith('queries'):
                  mine_task.queries_path=file
             if file.startswith('corpus'):
                  mine_task.collection_path=file 
             else:
                  mine_task.tripels=file
        
        mine_task.status='COMPLETED'
        mine_task.save()

    except Exception as e:
        logger.error(f"Error mining negatives for task {task_id}: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=5)


@shared_task(bind=True)
def train_model(self,task_id,learning_rate=1e-05,num_epochs=1):
    
    try:
        
        mine_task=NegativeMiningResult.objects.get(task_id=task_id)
        task_id=mine_task.task_id
        if mine_task.status!='COMPLETED':
            raise Exception(f'mining for {mine_task.task_id} is not complete yet')

        triples=mine_task.tripels
        query_path=mine_task.queries_path
        collection_path=mine_task.collection_path
        model_checkpoint=mine_task.model_checkpoint

        status='PROCESSING'
        training_task=RetrievalModel.objects.create(
            task_id=task_id,
            status=status
        )
        training_task.save()

        loss_path,checkpoint_paths=train(query_path=query_path,
                                         collection_path=collection_path,
                                         model_checkpoint=model_checkpoint,
                                         triples=triples,
                                         lr=learning_rate,
                                         num_epochs=num_epochs)
        
        training_task.loss_path=loss_path
        training_task.checkpoint_paths=str(checkpoint_paths)
        training_task.status='COMPLETED'
        training_task.save()

        logging.info("Pushing to HuggingFace")
        api=HfApi(login=HUGGINFACE_API_KEY)

        api.upload_folder(
        folder_path=checkpoint_paths[-1],
        repo_id="xorsuyash/upload_prac",
        repo_type="model",
        )
        

    except Exception as e:
        logger.error(f"Error in training for task {task_id}: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=5)
         
        



    
        






