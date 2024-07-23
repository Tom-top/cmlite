# cmlite

**cmlite** is a Python toolkit that simplifies/extends some of the functionalities of the [**ClearMap**](https://github.com/ChristophKirst/ClearMap2) pipeline.

### INSTALLATION
#### WINDOWS
1)	Install [**Git**](https://git-scm.com/downloads).
2)	Install [**Git LFS**](https://git-lfs.com/). This will be important to pull some of the larger files in the project: reference, atlas…
3)	Install [**Anaconda**](https://www.anaconda.com/download).
4)	Install [**Microsoft Visual C++ 14.0 or greater**](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
5)	Install an **integrated development environment**. I personally recommend [**Pycharm**](https://www.jetbrains.com/pycharm/download/?section=windows).
6)	Clone the **cmlite** repository in the PycharmProjects folder (located by default under Users\user_name) by running:
```
cd Users\user_name\PycharmProjects
git clone https://github.com/Tom-top/cmlite
```
7)	Create a conda environment using the environment.yml file by running:
```
conda env create -n cmlite --file Users\user_name\PycharmProjects\cmlite\environment.yml
```
8)	[OPTIONAL] Open Pycharm and open **Settings** (Shift + Cntrl + S). Look for **Python Console** in the search bar and add the following line of code to **Strating Script**:
```
%load_ext autoreload
%autoreload 2
```
This will reload modules before executing user code which is a significant quality of life improvement.

## Authors

* **Thomas TOPILKO**

## License

N.A