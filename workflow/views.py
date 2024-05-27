from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .serializers import CSVSerializer
from .celery_tasks import process_dataset,mine_negatives,train_model
from .models import DatasetProcessingTask,NegativeMiningResult,RetrievalModel
import uuid
import os
import logging
import datetime

logger = logging.getLogger(__name__)

def upload_pdf():
    pass 

class GenerateDataset:
    pass 


class ProcessDatasetView(APIView):

    def post(self, request, *args, **kwargs):
        serializer = CSVSerializer(data=request.data)

        if serializer.is_valid():
            try:
                dataset = serializer.validated_data['csv_file']

                dataset_name = f'datasets/{uuid.uuid4()}_{dataset.name.split(".")[0]}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv'

                dataset_path = default_storage.save(dataset_name, ContentFile(dataset.read()))

                os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

                
                task_id = str(uuid.uuid4())

                task=DatasetProcessingTask.objects.create(
                    task_id=task_id,
                    dataset_path=dataset_path
                )
                task.save()

                process_dataset.apply_async(args=[task.pk])

                return Response({'task_id': task_id}, status=status.HTTP_202_ACCEPTED)

            except Exception as e:
                logger.error(f"Error saving dataset: {str(e)}")

                return Response({'error': 'Failed to process dataset'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class MineNegativesView(APIView):

    def post(self, request, *args, **kwargs):
        task_id = request.query_params.get('task_id')
        model_name = request.data.get('model_name')
        model_checkpoint = request.data.get('model_checkpoint')

        if not task_id:
            return Response({'error': 'task_id is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            task = DatasetProcessingTask.objects.get(task_id=task_id)
        except DatasetProcessingTask.DoesNotExist:
            return Response({'error': 'task_id does not exist'}, status=status.HTTP_404_NOT_FOUND)

        if task.status != 'COMPLETE':
            return Response({'error': 'dataset processing is not yet complete'}, status=status.HTTP_400_BAD_REQUEST)

        if not model_name or not model_checkpoint:
            return Response({'error': 'model_name and model_checkpoint are required'}, status=status.HTTP_400_BAD_REQUEST)
        
        mining_result=NegativeMiningResult.objects.create(
            task_id=task_id,
            model_name=model_name,
            model_checkpoint=model_checkpoint
        )
        mining_result.save()

        mine_negatives.apply_async(args=[task.pk])

        return Response({'status': 'processing'}, status=status.HTTP_202_ACCEPTED)

class MineStatus(APIView):
    def get(self,request,*args,**kwargs):

        task_id=request.query_params.get('task_id')
        task=NegativeMiningResult.objects.get(task_id=task_id)

        if task.status=='PROCESSING':

            return Response({'task_id':task_id,'status':'Mining'},status=status.HTTP_202_ACCEPTED)
        
        if task.status=='COMPLETED':

            return Response({'task_id':task_id,'status':'Mining completed'},status=status.HTTP_202_ACCEPTED) 


class ModelTrainingView(APIView):

    def post(self,request,*args,**kwargs):

        task_id=request.query_params.get('task_id')

        if request.data.get('num_epochs'):
            num_epochs=int(request.data.get('num_epochs'))
        
        if request.data.get('lr'):
            learning_rate=float(request.data.get('lr'))

        if not task_id:
            return Response({'error': 'task_id is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            task = DatasetProcessingTask.objects.get(task_id=task_id)
        except DatasetProcessingTask.DoesNotExist:
            return Response({'error': 'task_id does not exist'}, status=status.HTTP_404_NOT_FOUND)
        
        train_model.apply_async(args=[task_id],kwargs={'learning_rate':learning_rate,'num_epochs':num_epochs})

        return Response({'status': 'training'}, status=status.HTTP_202_ACCEPTED)
        

class TrainingStatus(APIView):
    
    def get(self,request,*args,**kwargs):

        task_id=request.query_params.get('task_id')
        task=RetrievalModel.objects.get(task_id=task_id)

        if task.status=='PROCESSING':

            return Response({'task_id':task_id,'status':'Training'},status=status.HTTP_202_ACCEPTED)
        
        if task.status=='COMPLETED':

            return Response({'task_id':task_id,'status':'Completed'},status=status.HTTP_202_ACCEPTED)

     
        
        





