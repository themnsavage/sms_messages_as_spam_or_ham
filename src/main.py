from data_manager import Data_Manager


def main():
    file = "SMSSpamCollection"
    data_manager = Data_Manager(file=file)

    model = data_manager.make_model_with_training_data(max_models=50)
    data_manager.save_model(model=model, path="model")
    print(data_manager.evaluate_performance(model=model))
    
    # model = data_manager.use_saved_model(path='model/StackedEnsemble_BestOfFamily_1_AutoML_3_20231122_200845')
    # print(data_manager.evaluate_performance(model))


if __name__ == "__main__":
    main()
