To compile and run the code on cims cuda machines 2-5 (cuda1 is having problems).
	1. `ssh` into the cuda machine you wish to run the code on.
	2. Load the appropriate cuda module (cuda2/3/4 -> cuda-12.6; cuda5 -> cuda-13.0)
	3. Execute the `make` command for the machine you are on. E.g. on cuda3 run `$ make cuda3`
	4. To run the program with standard configurations, use command `$ make run`
		- Use your own configurations by passing the arguments `B=__ nh=__ N=__ d=__` into the `make run` command
	5. use `$ make clean` command to clean up executables, temporary files, etc.