from gtd.io import Workspace, sub_dirs
import re
from textmorph import data

def setup_exps_workspace(workspace_folder):
    exps_workspace = Workspace(getattr(data.workspace, workspace_folder))
    if len(sub_dirs(exps_workspace.root)) == 0:
        exp_folder_name = "exp_" + str(0)
    else:
        exp_num = max([int(re.search('(\d+)$', sub_dir_path).group(0)) for sub_dir_path in sub_dirs(exps_workspace.root)]) + 1
        exp_folder_name = "exp_"+str(exp_num)
    exps_workspace.add_dir(exp_folder_name, exp_folder_name)
    return Workspace(getattr(exps_workspace, exp_folder_name))
