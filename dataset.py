from datasets.Brain4cars import Brain4cars_Inside, Brain4cars_Outside, GazeDataset


def get_training_set_inside(opt, spatial_transform, horizontal_flip, temporal_transform,
                     target_transform):
    
    # assert opt.dataset_inside in ['Brain4cars_Inside', 'Brain4cars_Outside']


    training_data_inside = Brain4cars_Inside(
            opt.video_path_inside,
            opt.annotation_path,
            'training',
            opt.n_fold,
            opt.end_second,
            10,
            spatial_transform=spatial_transform,
            horizontal_flip=horizontal_flip,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data_inside

def get_training_set_outside(opt, spatial_transform, horizontal_flip, temporal_transform,
                     target_transform):
    
    # assert opt.dataset_outside in ['Brain4cars_Inside', 'Brain4cars_Outside']
    training_data_outside = Brain4cars_Outside(
        opt.video_path_outside,
        opt.annotation_path,
        'training',
        opt.n_fold,
        opt.end_second,
        10,
        spatial_transform=spatial_transform,
        horizontal_flip=horizontal_flip,
        temporal_transform=temporal_transform,
        target_transform=target_transform)

    return training_data_outside

def get_validation_set_inside(opt, spatial_transform, temporal_transform,
                       target_transform):
    # assert opt.dataset_inside in ['Brain4cars_Inside', 'Brain4cars_Outside']

    validation_data_inside = Brain4cars_Inside(
            opt.video_path_inside,
            opt.annotation_path,
            'validation',
            opt.n_fold,
            opt.end_second,
            opt.n_val_samples,
            spatial_transform,
            None,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration_inside)

    return validation_data_inside


def get_validation_set_outside(opt, spatial_transform, temporal_transform,
                       target_transform):
    # assert opt.dataset_outside in ['Brain4cars_Inside', 'Brain4cars_Outside']


    validation_data_outside = Brain4cars_Outside(
        opt.video_path_outside,
        opt.annotation_path,
        'validation',
        opt.n_fold,
        opt.end_second,
        opt.n_val_samples,
        spatial_transform=spatial_transform,
        horizontal_flip=None,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration_outside)

    return validation_data_outside


def get_training_set_gaze(opt):
    training_data_gaze = GazeDataset(
        opt.annotation_path,
        'training',
        opt.n_fold)

    return training_data_gaze

def get_validation_set_gaze(opt):
    validation_data_gaze = GazeDataset(
        opt.annotation_path,
        'validation',
        opt.n_fold)

    return validation_data_gaze





def get_testing_set_inside(opt, spatial_transform, temporal_transform,
                       target_transform):
    # assert opt.dataset_inside in ['Brain4cars_Inside', 'Brain4cars_Outside']

    test_data_inside = Brain4cars_Inside(
            opt.test_video_path_inside,
            opt.test_annotation_path,
            'test',
            # opt.n_fold,
            1,
            opt.end_second,
            opt.n_val_samples,
            spatial_transform,
            None,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration_inside)

    return test_data_inside


def get_testing_set_outside(opt, spatial_transform, temporal_transform,
                       target_transform):
    # assert opt.dataset_outside in ['Brain4cars_Inside', 'Brain4cars_Outside']


    test_data_outside = Brain4cars_Outside(
        opt.test_video_path_outside,
        opt.test_annotation_path,
        'test',
        # opt.n_fold,
        1,
        opt.end_second,
        opt.n_val_samples,
        spatial_transform=spatial_transform,
        horizontal_flip=None,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration_outside)

    return test_data_outside