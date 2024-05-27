from django.db import models

# Create your models here.
class DatasetProcessingTask(models.Model):
    task_id=models.CharField(max_length=255,unique=True)
    dataset_path=models.CharField(max_length=255)
    status=models.CharField(max_length=255,default='Pending')
    created_at=models.DateTimeField(auto_now_add=True)
    updates_at=models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.task_id} ({self.status})"

class NegativeMiningResult(models.Model):
    task_id=models.CharField(max_length=255,unique=True)
    model_name=models.CharField(max_length=255)
    model_checkpoint=models.CharField(max_length=255)
    tripels=models.CharField(max_length=255)
    queries_path=models.CharField(max_length=255)
    collection_path=models.CharField(max_length=255)
    status=models.CharField(max_length=255)
    created_at=models.DateTimeField(auto_now_add=True)

class RetrievalModel(models.Model):

    task_id=models.CharField(max_length=255,unique=True)
    loss_path=models.CharField(max_length=255,default='default')
    checkpoint_paths = models.CharField(max_length=2000, default='default')
    status=models.CharField(max_length=255,default='Training')

    

    