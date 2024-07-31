# cmlite

**cmlite** is a Python toolkit that simplifies/extends some of the functionalities of the [**ClearMap**](https://github.com/ChristophKirst/ClearMap2) pipeline.

### INSTALLATION
#### WINDOWS
1) Install [**Git**](https://git-scm.com/downloads).
2) Install [**Anaconda**](https://www.anaconda.com/download).
3) Install [**Microsoft Visual C++ 14.0 or greater**](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
4) Install an **integrated development environment**. I personally recommend [**Pycharm**](https://www.jetbrains.com/pycharm/download/?section=windows).
5) Clone the **cmlite** repository in the PycharmProjects folder (located by default under Users\user_name) by running:
```
cd Users\user_name\PycharmProjects
git clone https://github.com/Tom-top/cmlite
```
6) Download large files related to the projects [**here**](https://www.dropbox.com/scl/fo/ld9re7620kela1oovblsj/ADDly_yw2M0huvf-bS1DsfE?rlkey=5ctl8a5sdc882nu8hesrx9ca9&st=l8nz9121&dl=0). Extract the folders and place them under "cmlite\ressources".
7) Create a conda environment using the environment.yml file by running:
```
conda env create -n cmlite --file Users\user_name\PycharmProjects\cmlite\environment.yml
```
8) [OPTIONAL] Open Pycharm and open **Settings** (Shift + Cntrl + S). Look for **Python Console** in the search bar and add the following line of code to **Strating Script**:
```
%load_ext autoreload
%autoreload 2
```
This will reload modules before executing user code which is a significant quality of life improvement.

## Authors

* **Thomas TOPILKO**

## License

N.A