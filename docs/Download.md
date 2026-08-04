If you are reading this documentation, the easiest way to download AthenaK is to fork it directly using the link on the [repository](https://github.com/IAS-Astrophysics/athenak) page.

Else, you can clone a copy from the command line using any git client. Let the target directory for the code be denoted by `${athena}`.  Then clone with:

    $ git clone --recursive https://github.com/IAS-Astrophysics/athenak.git ${athena}

If you are a developer, you may wish to clone using two-factor authentication on GitHub using

    $ git clone --recursive https://USERNAME:TOKEN@github.com/IAS-Astrophysics/athenak.git ${athena}

where `TOKEN` is your [GitHub personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) for your `USERNAME`.  Else you will have to enter your credentials every time you try to make a commit.

The `--recursive` option clones the Kokkos repository along with AthenaK.  If you clone the code without Kokkos, you must install it manually in the top-level directory containing the code:

    $ git clone https://github.com/IAS-Astrophysics/athenak.git $athena
    $ cd ${athena}
    $ git submodule init
    $ git submodule update