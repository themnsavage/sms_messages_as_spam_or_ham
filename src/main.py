from data_manager import Data_Manager


def main():
    file = "SMSSpamCollection"
    data_manager = Data_Manager(file=file)
    print(f"data set head:\n{data_manager.get_data_set_head()}")
    print("\n\n\n")

    feature_train, label_train = data_manager.get_train_data_set()
    print(f"data set train feature:\n {feature_train.head()}\n")
    print(f"data set train label:\n {label_train.head()}")
    print("\n\n\n")

    feature_test, label_test = data_manager.get_test_data_set()
    print(f"data set test feature:\n {feature_test.head()}\n")
    print(f"data set test label:\n {label_test.head()}")
    print("\n\n\n")


if __name__ == "__main__":
    main()
