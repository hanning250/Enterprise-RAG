"""绝对路径工具模块"""

import os


def get_project_root() -> str:
    """获取项目根目录的绝对路径。"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(current_dir)
    return project_root


def get_abs_path(relative_path: str) -> str:
    """将相对于项目根目录的相对路径拼接成绝对路径。"""
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)


if __name__ == '__main__':
    print(get_abs_path("config/rag.yml"))
