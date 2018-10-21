# GIF Sentiment Analysis

## Introduction
GIF's are an incredibly common way for people to express ideas, sentiments, and emotions to each other.
Whether the GIF contains a goat prancing around, Homer Simpson receding into a hedge, or someone
burying their face into their palm, we as humans have no problem understanding the sentiment.

Can the same be said for computers?

## Our Project
From a technical perspective, this is an incredibly challenging problem. The content in GIF's span nearly every type 
of media and visualization found on the internet -- everything is fair game. Almost no assumptions can be made 
regarding a GIF: they have a varying durations, may contain text, physics defying special FX, and usually incredibly 
lossy image compression to top it off.

Nevertheless, our project is an attempt to solve this problem by using deep learning and leveraging 
the crowd-sourced data from the MIT GIFGIF Media Lab. As part of their project, they present users with two GIF's and 
allow them to choose which one best represents a given sentiment. In this manner, they have generated a
labeled data-set with 6,000+ unique samples.

## Our Approach
Instead of trying to tackle the problem head on, we've decided to break it down into segments, and to individually
address those to form a complete solution. Some things we are working on:
* Detecting people, and extracting features from the progression of their facial expressions.
* Summarization techniques; finding ways to effectively represent a sequence of images as a single image.
* Correlation between sentiments and the color features of the GIF, e.g. quantifying how a dark and gloomy tone
of an image affects the perceived sentiment.

## Example GIF's

![GIF Slideshow](media/slow_gif_slideshow.gif)