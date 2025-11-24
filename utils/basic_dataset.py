def get_data_range(flag, total_len, seq_len):
    # divide data into train, vali, finetune, test parts
    num_train = int(total_len * 0.6)
    num_finetune = int(total_len * 0.1)
    num_test = int(total_len * 0.2)
    num_vali = total_len - num_train - num_finetune - num_test

    # get borders of the data
    if flag == 'train':  # train
        border1 = 0
        border2 = num_train
    elif flag == 'val':  # val
        border1 = num_train - seq_len
        border2 = num_train + num_vali
    elif flag == 'finetune':  # finetune
        border1 = num_train + num_vali - seq_len
        border2 = num_train + num_vali + num_finetune
    elif flag == 'test':  # test
        border1 = total_len - num_test - seq_len
        border2 = total_len
    else:
        raise NotImplementedError
    return border1, border2, num_train
