from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("data") #训练的目录
model_trainer.trainModel(num_objects=4, num_experiments=50, enhance_data=True, batch_size=5, show_network_summary=True)
