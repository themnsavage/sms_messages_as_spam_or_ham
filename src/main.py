from model_manager import Model_Manager


def main():
    file = "SMSSpamCollection"
    model_manager = Model_Manager(file=file)

    # model = model_manager.make_model_with_training_data(max_models=4)
    # model_manager.save_model(model=model, path="model")
    # print(model_manager.evaluate_performance(model=model))
    
    model = model_manager.use_saved_model(path='model/GLM_1_AutoML_1_20231122_221751')
    # print(model_manager.evaluate_performance(model))
    # print(model_manager.make_predictions(model=model))
    model_manager.make_single_prediction(model=model)

if __name__ == "__main__":
    main()
