import os
import git
import yaml
import logging
import time
logger = logging.getLogger('GIT')

## TODO need to figure out auto-push when SSH keys are not active... (both normally and in container)

git_enabled = not os.environ.get('IS_CONTAINER', False)
whoami = os.environ.get('USER', 'guest')

local_dev = f'{whoami}-auto-dev'
local_run = f'{whoami}-auto-run'
version_file = 'src/auto/repo/version.yml'
repo_path = '.'

skip_checks = ['src/auto/system_parameters.py', version_file]
_checks = ['src/ml/models/', 'src/ml/models/layers/', 'src/ml/models/connections/', 'src/ml/models/blocks/', 'src/ml/models/act/', 'src/ml/param/', 'src/ml/runners/', 'src/ml/iox/']
check = lambda x: any([s in x for s in _checks]) 

def repo(strict=True):
    os.system('wandb login')
    try:
        repo = git.Repo(repo_path)
        if local_dev not in repo.branches:
            checkout_local_dev_branch(repo)
        if strict:
            on_dev(repo)
            if repo.is_dirty():
                logger.warning(f"Repository is dirty. Committing changes.")
                repo.git.add(all=True)
                msg = input("MSG: ")
                repo.git.commit(message=f'[auto] {msg}')
        return repo
    except (git.exc.InvalidGitRepositoryError):
        repo = git.Repo.init(None)
        repo.config_writer().set_value('push', 'followTags', 'true').release()
        checkout_local_dev_branch(repo)
        return repo

# Make sure you have a repository, and there is a remote named {remote_name} and a branch named {branch_name}.
# The easiest way to fix this is to clone the repository (from the appropriate tagged release).""")

def on_dev(repo):
    assert local_dev == repo.head.ref.name, f"Expected to be on {local_dev}, but on {repo.head.ref.name}. Switch first."
        
def checkout_local_dev_branch(repo):
    local_dev = f'{whoami}-auto-dev'
    if local_dev not in repo.branches:
        repo.create_head(local_dev)
    repo.git.checkout(local_dev) 
    
def source_modified(repo, expected_commit_hash):
    # check if local_dev is ahead of local_run
    if local_run not in repo.branches:
        return True
    commit1 = repo.heads[local_dev].commit
    commit2 = repo.heads[local_run].commit
    
    # check if commit hash has changed
    commit_hash = repo.heads[local_dev].commit.hexsha
    commit_hash_changed = commit_hash != expected_commit_hash
    
    # check diffs
    diff = repo.git.diff(name_only=True).split('\n') # get diff against index
    diff = [d for d in diff if check(d)]
    logger.info(f"Source diff: {diff}")

    # create state modified flags
    smb = (len(diff) > 0) or commit_hash_changed # source modified
    sm = commit1 != commit2 or smb              # source modified strict (incl. run commit)
    
    return sm, smb, commit_hash
    
def load_version(f):
    version_data = yaml.load(open(f).read(), Loader=yaml.FullLoader)
    return version_data

def save_version(f, version_data):
    with open(f,'w') as file:
        yaml.dump(version_data, file)
       
def remove_index_lock():
    if os.path.exists('.git/index.lock'):
        logger.warn('Automatically removing .git/index.lock (in 2s); please make sure no other git process is running.')
        time.sleep(2)
        os.remove('.git/index.lock')    
        
def tag_and_version(repo, tag=True):
    if local_run not in repo.branches:
        repo.create_head(local_run)
        
    version_data = load_version(version_file)    
    expected_commit_hash = version_data['commit']
    sm, smb, commit_hash = source_modified(repo, expected_commit_hash)
        
    if (sm and tag) or smb:
        version = version_data['version']
        version[2] += 1
        version_data['version'] = version
        version_data['commit'] = commit_hash
        version_data['action'] = 0
    # update task number
    else:
        version_data['action'] += 1
    save_version(version_file, version_data)
        
    try:        
        repo.git.add('-u') # update index for next diff check
        remove_index_lock() # remove again just in case
    except Exception as e:
        remove_index_lock()
        repo.git.add('-u')
        remove_index_lock() # remove again just in case    
    
    if tag:
        try:
            if sm:
                repo.git.add(version_file)
                repo.git.commit(message=f'[auto] version update')
                repo.git.checkout(local_run)
                repo.git.merge(local_dev)
                tag = '.'.join([str(v) for v in version])
                repo.create_tag(tag)
                repo.git.checkout(local_dev)
        except Exception as e:
            logger.error(f"Failed to update git and tag: {e}")
            repo.git.checkout(local_dev) 
    return version_data, sm

def get_save_path(version_data, strict, action_name, base_save_path):
    vs = [str(v) for v in version_data['version']]
    version = f'v-{".".join(vs)}{"" if strict else "a"}'
    fname =  f'{version}-{version_data["action"]}-{action_name}'
    sp = os.path.join(base_save_path, fname)
    logger.info(f'Save path: {sp}')
    os.makedirs(sp, exist_ok=True)
    return version, str(version_data["action"]), sp

def save_env(v, a, sp):
    env = {
        'version': v,
        'action': a,
        'save_path': sp
    }
    os.environ.update(env)
    

# parameters.z_tstat = time_stat(Weierstrass)
# 