# Generated by Django 5.0.6 on 2024-05-27 03:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('workflow', '0002_negativeminingresult_model_checkpoint_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='negativeminingresult',
            name='task_id',
            field=models.CharField(max_length=255, unique=True),
        ),
    ]
