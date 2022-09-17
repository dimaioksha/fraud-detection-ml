class F1BinaryEvaluator():

    def __init__(self, predCol="prediction", labelCol="label", metricLabel=1.0):
        self.labelCol = labelCol
        self.predCol = predCol
        self.metricLabel = metricLabel

    def isLargerBetter(self):
        return True

    def evaluate(self, dataframe):
        tp = dataframe.filter(self.labelCol + ' = ' + str(self.metricLabel) + ' and ' + self.predCol + ' = ' + str(self.metricLabel)).count()
        fp = dataframe.filter(self.labelCol + ' != ' + str(self.metricLabel) + ' and ' + self.predCol + ' = ' + str(self.metricLabel)).count()
        fn = dataframe.filter(self.labelCol + ' = ' + str(self.metricLabel) + ' and ' + self.predCol + ' != ' + str(self.metricLabel)).count()
        return tp / (tp + (.5 * (fn +fp)))


