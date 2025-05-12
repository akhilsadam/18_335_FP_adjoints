system={
    'actions':
    [
        {
            'action_name': 'burgers-sanity-check',#
            'no_compute': False,
            'strict_version_checks': False,
            'debug': False,
            'save_frequency': 100,
            'ngpu': 1,
            'data': "data/VB/",
            'tasks':
            [                
                {
                    'task_name': 'sanity_check',
                    'task_parameters': {}, # will scan over these if not empty
                    'task_type': 'runners.train',
                    'run_parameters': {
                        'model': 'models.sanity_check',
                        'model_parameters': {
                        },
                        'loader_parameters': {
                            'num_workers': 7,
                            'train_memory_length': 1,
                            'test_memory_length': 1,
                            'train_predict_length': 1,
                            'test_predict_length': 1,
                        },
                    }
                }
             
            ]
        },
     ]
}