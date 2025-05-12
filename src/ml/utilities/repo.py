import os
def understand_env():
    return (
        os.environ.get('version', '0.0.0'),
        os.environ.get('action', 0),
        os.environ.get('save_path', 'save')
    )