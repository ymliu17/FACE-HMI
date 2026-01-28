from trainer.ft_base import FACE_Trainer

def make_FACE_trainer(**kwargs):
    print("Creating trainer...")

    assert 'model' in kwargs, "model is required"
    assert 'optimizer' in kwargs, "optimizer is required"
    assert 'criterion' in kwargs, "criterion is required"
    assert 'train_loader' in kwargs, "train_loader is required"
    assert 'test_loader' in kwargs, "test_loader is required"
    assert 'epochs' in kwargs, "epochs is required"
    assert 'log_dir' in kwargs, "log_dir is required"
    assert 'target' in kwargs, "target is required"


    return FACE_Trainer(**kwargs)