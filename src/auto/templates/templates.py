import os
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('src/auto/templates'),comment_start_string='###',line_statement_prefix = '###',line_comment_prefix='###')

script_folder = 'src/auto/tmp'
sg_engine_filename = 'src/auto/tmp/sg_engine.sh'

init = 'src/auto/init'
install = 'src/auto/install'
run_script = 'run.sh'

def sg_engine(**kwargs):
    template = env.get_template('sg_engine.jinja')
    return template.render(**kwargs)


def write_script(filename, script):
    os.makedirs(script_folder, exist_ok=True)
    with open(filename, 'w') as f:
        f.write(script)
    os.system(f'chmod +x {filename}')


def single_run(run, runner, cnf, logfile):
    template = env.get_template('single_run.jinja')
    script = template.render(init=init, install=install, run=run, runner=runner, configfile=cnf, logfile=logfile)
    return script
    

class Bash():
    def __init__(self, filename, script, delete=False, **kwargs):
        self.filename = filename
        self.script = script
        self.delete = delete
        self.kwargs = kwargs
    def __enter__(self):
        write_script(self.filename, self.script)
    
    def __exit__(self, *args, **kwargs):
        if self.delete:
            self.delete_script()
    
    def delete_script(self):
        assert self.filename is not None
        assert os.path.exists(self.filename)
        assert self.filename.contains('/tmp/')
        sleep(4)
        os.system(f'rm {self.filename}')