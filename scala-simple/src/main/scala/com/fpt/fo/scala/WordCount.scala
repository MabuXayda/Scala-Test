package com.fpt.fo.scala

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object WordCount {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("WordCount").setMaster("local")
    val sc = new SparkContext(conf)
    val text = sc.wholeTextFiles("./data/docsvn/*.txt").map(_._2)
    val wordsCount = text.flatMap(line => line.split("\\s")).map(word => (word, 1)).reduceByKey(_ + _)

    wordsCount.saveAsTextFile("./data/wordCount")
  }
}