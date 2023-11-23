from model_manager import Model_Manager


def main():
    file = "SMSSpamCollection"
    model_manager = Model_Manager(file=file)

    # model = model_manager.make_model_with_training_data(max_models=35)
    # model_manager.save_model(model=model, path="model")
    # print(model_manager.evaluate_performance(model=model))
    # model_manager.print_pearson_correlation()

    model = model_manager.use_saved_model(path='model/GBM_3_AutoML_1_20231123_10521')
    print(model_manager.evaluate_performance(model=model))
    print(model_manager.make_predictions(model=model))
    print(model_manager.make_single_prediction(model=model,single_message="Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"))


if __name__ == "__main__":
    main()
