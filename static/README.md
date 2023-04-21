# Introduction

In my project Mustuart, I attempted to package my app for WASM using Pyodide, a Python runtime environment for running Python in the browser. However, I ran into an issue with the `python_speech_features` package, which did not have a pre-built Python wheel available on the PyPI package repository. To solve this issue, I built my own Python wheel for the package, which allowed me to package the app successfully. 

***It is important to note that I do not own the `python_speech_features` package.***

During the process, I followed the license terms of the package, which allowed for modification and distribution, and credited the original authors in the package's metadata. Additionally, I made sure to clearly state in my project's README and documentation that the `python_speech_features` package is not owned by me, and gave credit to the original authors.
