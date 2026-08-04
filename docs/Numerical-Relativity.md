# Running Binary Black Hole

AthenaK requires external modules to provide Cauchy initial date for the binary black hole problems. Currently, we can read in initial data from TWOPUNCTURE and SpECTRE code. The TWOPUNCTURE initial data is calculated on the fly at the start of the evolution, while SpECTRE initial data needs to be pre-computated. Below is a tutorial for evolving the TWOPUNCTURE initial data. 


## TWOPUNCTURE Initial Data
Two external libraries are required for building the z4c_two_puncture problem, i.e. `gsl` and `twopuncturesc` initial data solver (in this order). 

```
cd $HOME && mkdir -p usr/gsl && mkdir codes && cd codes
# grab source
wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.5.tar.gz

# extract and configure for local install
tar -zxvf gsl-2.5.tar.gz

cd gsl-2.5

./configure --prefix=$HOME/usr/gsl

make -j8
# make check
make install

# link gsl into athenak
ln -s $HOME/usr/gsl ${path_to_athenak}
```
#### twopuncturesc:
```
cd $HOME && cd usr
git clone https://github.com/computationalrelativity/TwoPuncturesC.git
cd TwoPuncturesC
make -j8

# link twopuncturesc into athenak
ln -s $HOME/usr/TwoPuncturesC ${path_to_athenak}
```
Now create build directory and configure with cmake

```mkdir build_z4c_twopunc && cd build_z4c_twopunc```

Finally, we can configure and build Athenak

$ cmake ../ -DPROBLEM=z4c_two_puncture


(TO DO (HZ): add tutorial for SpECTRE ID)