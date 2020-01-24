---
title: 在Linux上安装Homebrew时遇到的一些问题
urlname: 在Linux上安装Homebrew时遇到的一些问题
toc: true
date: 2019-04-23 00:59:31
updated: 2019-04-23 00:59:31
tags:
categories:
---

最后我还是安装大失败了，不打算再去管它。

<!--more-->

虽然我在实验室的服务器上没有sudo权限，但仍然会想要安装一些包，但是自己手动管理所有依赖和源码编译实在太麻烦了。

按照上述步骤做完之后，发现实际上brew没装好，虽然输入`brew`可以打出来正常的结果：

```sh
~$ brew
Example usage:
  brew search [TEXT|/REGEX/]
  brew info [FORMULA...]
  brew install FORMULA...
  brew update
  brew upgrade [FORMULA...]
  brew uninstall FORMULA...
  brew list [FORMULA...]

Troubleshooting:
  brew config
  brew doctor
  brew install --verbose --debug FORMULA

Contributing:
  brew create [URL [--no-fetch]]
  brew edit [FORMULA...]

Further help:
  brew commands
  brew help [COMMAND]
  man brew
  https://docs.brew.sh
```

但无论`brew install`什么都会报错：

```sh
~$ brew install fish
==> Installing dependencies for git: glibc, libmpc, isl@0.18, gcc, bzip2, pcre2, libbsd and expat
glibc: gawk is required to build glibc.
Install gawk with your host package manager if you have sudo access.
  sudo apt-get install gawk
  sudo yum install gawk
Error: An unsatisfied requirement failed this build.
==> Installing dependencies for fish: glibc, libmpc, isl@0.18, gcc, bzip2 and pcre2
glibc: gawk is required to build glibc.
Install gawk with your host package manager if you have sudo access.
  sudo apt-get install gawk
  sudo yum install gawk
Error: An unsatisfied requirement failed this build.
```

后来我尝试了一下这种安装gawk的方法，它成功多下了几个包。

```sh
~$ ~$ brew deps --include-build gawk
gettext
gmp
gpatch
m4
mpfr
ncurses
pkg-config
readline
zlib
~$ brew install --ignore-dependencies $(brew deps --include-build gawk)
==> Installing dependencies for git: glibc, m4, gmp, mpfr, libmpc, isl@0.18, gcc, gpatch, ncurses, gettext, bzip2, pcre2, libbsd and expat
glibc: gawk is required to build glibc.
Install gawk with your host package manager if you have sudo access.
  sudo apt-get install gawk
  sudo yum install gawk
Error: An unsatisfied requirement failed this build.
Warning: --ignore-dependencies is an unsupported Homebrew developer flag!
Adjust your PATH to put any preferred versions of applications earlier in the
PATH rather than using this unsupported flag!

Warning: pkg-config 0.29.2_1 is already installed and up-to-date
To reinstall 0.29.2_1, run `brew reinstall pkg-config`
Warning: zlib 1.2.11 is already installed and up-to-date
To reinstall 1.2.11, run `brew reinstall zlib`
==> Downloading https://ftp.gnu.org/gnu/gettext/gettext-0.19.8.1.tar.xz
######################################################################## 100.0%
==> Downloading http://xmlsoft.org/sources/libxml2-2.9.7.tar.gz
######################################################################## 100.0%
==> ./configure --prefix=/home/zhm/.linuxbrew/Cellar/gettext/0.19.8.1_1/libexec --without-python --without-lzma
==> make install
==> ./configure --disable-silent-rules --prefix=/home/zhm/.linuxbrew/Cellar/gettext/0.19.8.1_1  --with-included-glib --with-inc
==> make
==> make install
🍺  /home/zhm/.linuxbrew/Cellar/gettext/0.19.8.1_1: 2,198 files, 28MB, built in 6 minutes 1 second
==> Downloading https://linuxbrew.bintray.com/bottles/gmp-6.1.2_2.x86_64_linux.bottle.1.tar.gz
==> Downloading from https://akamai.bintray.com/09/09d722e6321b67257e80f08b7c69202b5898189ccd81677adcdc714faaa86e3b?__gda__=exp
######################################################################## 100.0%
==> Pouring gmp-6.1.2_2.x86_64_linux.bottle.1.tar.gz
🍺  /home/zhm/.linuxbrew/Cellar/gmp/6.1.2_2: 20 files, 3.8MB
==> Downloading https://linuxbrew.bintray.com/bottles/gpatch-2.7.6.x86_64_linux.bottle.tar.gz
==> Downloading from https://akamai.bintray.com/70/70df1fb356ca7ccccd277110fcf467ea9fd0dc7337c93ad8ecd39848b081f95c?__gda__=exp
######################################################################## 100.0%
==> Pouring gpatch-2.7.6.x86_64_linux.bottle.tar.gz
🍺  /home/zhm/.linuxbrew/Cellar/gpatch/2.7.6: 10 files, 909.6KB
==> Downloading https://linuxbrew.bintray.com/bottles/m4-1.4.18.x86_64_linux.bottle.tar.gz
==> Downloading from https://akamai.bintray.com/5a/5a2327087fb76145b4d0fb23acc244115adc3ced14ffc2a6231159a4f16c8a7f?__gda__=exp
######################################################################## 100.0%
==> Pouring m4-1.4.18.x86_64_linux.bottle.tar.gz
🍺  /home/zhm/.linuxbrew/Cellar/m4/1.4.18: 13 files, 1.1MB
==> Downloading https://linuxbrew.bintray.com/bottles/mpfr-4.0.2.x86_64_linux.bottle.tar.gz
==> Downloading from https://akamai.bintray.com/cf/cffaa9976a516130ac5b979eae7bdd62180ed8924bb52bf0b30935174b1cbab0?__gda__=exp
######################################################################## 100.0%
==> Pouring mpfr-4.0.2.x86_64_linux.bottle.tar.gz
🍺  /home/zhm/.linuxbrew/Cellar/mpfr/4.0.2: 29 files, 5.2MB
==> Downloading https://ftp.gnu.org/gnu/ncurses/ncurses-6.1.tar.gz
######################################################################## 100.0%
==> ./configure --prefix=/home/zhm/.linuxbrew/Cellar/ncurses/6.1 --enable-pc-files --with-pkg-config-libdir=/home/zhm/.linuxbre
==> make install
🍺  /home/zhm/.linuxbrew/Cellar/ncurses/6.1: 3,856 files, 9.2MB, built in 1 minute 17 seconds
==> Downloading https://linuxbrew.bintray.com/bottles/readline-8.0.0_1.x86_64_linux.bottle.tar.gz
==> Downloading from https://akamai.bintray.com/ae/aecadfc325735c80de0965dd31292d34d2c3ed0cb404a7adb8288d6f99a78e94?__gda__=exp
######################################################################## 100.0%
==> Pouring readline-8.0.0_1.x86_64_linux.bottle.tar.gz
🍺  /home/zhm/.linuxbrew/Cellar/readline/8.0.0_1: 48 files, 1.9MB
```

然而坏掉的brew自己还是没办法install gawk……

```sh
brew install -s gawk
==> Installing dependencies for git: glibc, libmpc, isl@0.18, gcc, bzip2, pcre2, libbsd and expat
glibc: gawk is required to build glibc.
Install gawk with your host package manager if you have sudo access.
  sudo apt-get install gawk
  sudo yum install gawk
Error: An unsatisfied requirement failed this build.
==> Installing dependencies for gawk: glibc, libmpc, isl@0.18 and gcc
glibc: gawk is required to build glibc.
Install gawk with your host package manager if you have sudo access.
  sudo apt-get install gawk
  sudo yum install gawk
Error: An unsatisfied requirement failed this build.
```

我猜，错误的原因大概是这样的：安装fish（和其他很多软件）的时候需要git clone源代码，但是git还没编译好，git依赖于glibc，glibc又依赖于gawk——然后brew居然觉得gawk又依赖于glibc？（之前dependency里明明没有它的……）于是，这怎么可能安装好呢。

最后只好自己编译了一个gawk。然后似乎直接有bin还不行，必须要link好……（所以编译出来的bin和安装完的有何区别？）而且随便安装还不行，最后安装到`.linuxbrew/Cellar`文件夹下了……