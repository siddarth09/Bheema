import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'bheema'


def package_files(data_files, directory_list, destination_base):
    for directory in directory_list:
        for (path, directories, filenames) in os.walk(directory):
           
            rel_path = os.path.relpath(path, start=directory)
            dest_path = os.path.join(destination_base, directory, rel_path)
      
            files = [os.path.join(path, f) for f in filenames]
            
            
            if files:
                data_files.append((dest_path, files))
    return data_files


data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]


data_files.append(('share/' + package_name + '/launch', glob('launch/*.py')))
data_files.append(('share/' + package_name + '/config', glob('config/*.yaml')))


data_files = package_files(data_files, ['unitree_g1'], 'share/' + package_name)

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sid',
    maintainer_email='sid@todo.todo',
    description='Bheema MPC Controller',
    license='MIT',
   
    entry_points={
        'console_scripts': [
            'convex_mpc_ros2 = bheema.convex_mpc_ros2:main',
        ],
    },
)