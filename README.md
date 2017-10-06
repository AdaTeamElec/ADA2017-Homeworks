# Homeworks
Repo for homework

# Add Upstream

```
git remote -v
git remote add upstream https://github.com/epfl-ada/ADA2017-Homeworks.git
git remote -v
```
# Syncing a fork

Fetch the branches and their respective commits from the upstream repository. Commits to '''master''' will be stored in a local branch, '''upstream/master'''. Check out your fork's local '''master''' branch. Then merge branches.
```
git fetch upstream
git checkout master
git merge upstream/master
```

Original link to https://help.github.com/articles/syncing-a-fork/
