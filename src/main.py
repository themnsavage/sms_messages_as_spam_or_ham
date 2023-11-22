from data_manager import Data_Manager


def main():
    file = "SMSSpamCollection"
    data_manager = Data_Manager(file=file)
    print(data_manager.get_data_set_head())


if __name__ == "__main__":
    main()