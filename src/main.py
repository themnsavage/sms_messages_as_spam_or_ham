from model_manager import Model_Manager


def main():
    file = "SMSSpamCollection"
    model_manager = Model_Manager(file=file)

    model = model_manager.make_model_with_training_data(max_models=30)
    model_manager.save_model(model=model, path="model")
    print(model_manager.evaluate_performance(model=model))
    
    # model = model_manager.use_saved_model(path='model/StackedEnsemble_BestOfFamily_1_AutoML_3_20231122_200845')
    # print(model_manager.evaluate_performance(model))


if __name__ == "__main__":
    main()
