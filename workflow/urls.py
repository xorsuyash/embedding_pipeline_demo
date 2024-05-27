from django.urls import path 
from .views import ProcessDatasetView,MineNegativesView,MineStatus,ModelTrainingView,TrainingStatus


urlpatterns=[path('process/',ProcessDatasetView.as_view(),name='process-dataset'),
             path('mine_negatives/',MineNegativesView.as_view(),name='mine-negatives'),
             path('mining_status/',MineStatus.as_view(),name='mining-status'),
             path('colbert_finetuning/',ModelTrainingView.as_view(),name='model-training'),
             path('training-status/',TrainingStatus.as_view(),name='training-status')]