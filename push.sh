#!/bin/bash
while getopts m: flag
do
    case "${flag}" in
        m) commit=${OPTARG};;
    esac
done

export https_proxy=http://$claship:7890;export http_proxy=http://$claship:7890;export all_proxy=socks5://$claship:7891
git add .
git commit -m "$commit"
git push --force