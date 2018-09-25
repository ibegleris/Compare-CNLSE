# banded-nlse
[![Build Status](https://travis-ci.org/ibegleris/Compare-CNLSE.svg?branch=master)](https://travis-ci.org/ibegleris/Compare-CNLSE)

[![DOI](https://zenodo.org/badge/132150278.svg)](https://zenodo.org/badge/latestdoi/132150278)


This repository holds the codes used for comparing the Generalised nonlinear Schrodinger equation (GNLSE) and the new Banded nonlinear equation (BNLSE) in terms of frequency bands. 


* Requirements:
  * Tested on Ubuntu Xenian, Ubuntu Trusty and OSX, although should be fine on any Unix based system. Windows is NOT supported. 
  * Python 3.6 tested
  * (Optional but recommended) The Conda Intel Python distribution found [here](https://software.intel.com/en-us/articles/using-intel-distribution-for-python-with-anaconda)

* Installation (Assuming you have Python 3 installed)
  * Install packages: pip install -r requirements.txt

* Execution:
 	* chmod +x run.sh
 	* ./run.sh
 	* Parameters changed in inputs() within src/main.py

Reference Journal paper:

Ioannis Begleris and Peter Horak, "Frequency-banded nonlinear Schrödinger equation with inclusion of Raman nonlinearity," Opt. Express 26, 21527-21536 (2018)
