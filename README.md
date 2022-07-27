# BSC-DS-2022

A repository for all lecture, review, or other resources for the Flatiron School/Birmingham Southern College Summer 2022 Program!

## All Machine Learning All The Time

![model training versus testing, shared by @rishiyer on twitter](https://pbs.twimg.com/media/FXMtZKtUEAElh0V.jpg)

## Written Instructions to Connect to This Repository:

**We will go through and follow these instructions together in a lecture during Week 1 - no rush!** 

1. FORK this repository, creating a copy on your own GitHub account

2. Then clone your fork down to your local computer
```
git clone https://github.com/[yourusername]/BSC-DS-2022.git
```

3. Add the `/flatiron-school/` version as the `upstream` (to pull future changes)
```
git remote add upstream https://github.com/flatiron-school/BSC-DS-2022.git
```

Not necessarily a next step, but you can make changes to any files or notebooks now! You can then push your changes to your forked version of the repo (to put your local changes up online on your own GitHub account):
```
git add [filename]
git commit -m 'message here'
git push
```

### Whenever you want to get updated notes:

5. Grab the changes from the upstream repo
```
git fetch upstream
```

6. Merge new changes onto your local repo
```
git merge upstream/main -m "meaningful message about what you're updating"
```
