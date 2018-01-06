---
layout: post
title: Add MathJax Parsing for your Jkeyll
tags: [jkeyll, programming]
date: 2017-08-17 21:19:00 +08:00
---


**配置`_includes/head.html`文件**

在你的`head.html`文件中做以下配置以保证能够链接到Mathjax

```javascript
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
      }
    });
  </script>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script> 
```

说明：`inlineMath`参数的设置是为了告诉MathJax对于行内数学表达式的解析，因为默认情况下是使用`\(\)`来辨别行内数学表达式的，我这里改成了`$\Latex$`

然后就大功告成了, 详细资料可以阅读[Documentation for MathJax](http://docs.mathjax.org/en/latest/tex.html)
