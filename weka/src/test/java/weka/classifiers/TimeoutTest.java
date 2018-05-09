package weka.classifiers;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Test;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.lazy.LWL;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.AdditiveRegression;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.ClassificationViaRegression;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.meta.RandomCommittee;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.meta.RegressionByDiscretization;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.HoeffdingTree;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;

public class TimeoutTest {

  @Test
  public void testTimeout() throws FileNotFoundException, IOException, InterruptedException {
    Classifier[] classifiers = { new NaiveBayes(), new NaiveBayesMultinomial(), new GaussianProcesses(), new LinearRegression(), new Logistic(), new MultilayerPerceptron(),
        new SGD(), new SimpleLinearRegression(), new SimpleLogistic(), new SMO(), new VotedPerceptron(), new IBk(), new KStar(), new LWL(), new DecisionTable(), new JRip(),
        new OneR(), new PART(), new ZeroR(), new DecisionStump(), new HoeffdingTree(), new J48(), new LMT(), new RandomForest(), new RandomTree(), new REPTree(), new AdaBoostM1(),
        new AdditiveRegression(), new AttributeSelectedClassifier(), new Bagging(), new ClassificationViaRegression(), new LogitBoost(), new MultiClassClassifier(),
        new RandomCommittee(), new RandomSubSpace(), new Stacking(), new Vote(), new RegressionByDiscretization() };

    Instances data = new Instances(new FileReader(new File("../datasets/classification/multi-class/amazon.arff")));
    data.setClassIndex(data.numAttributes() - 1);

    for (Classifier c : classifiers) {
      AtomicInteger interrupted = new AtomicInteger(0);

      Thread t = new Thread() {
        @Override
        public void run() {
          System.out.println(Thread.currentThread().getName() + ": Start Training Classifier " + c.getClass().getName());
          try {
            c.buildClassifier(data);
          } catch (InterruptedException e) {
            interrupted.incrementAndGet();
          } catch (Exception e) {
            e.printStackTrace();
          }
          System.out.println(Thread.currentThread().getName() + ": Done Training Classifier " + c.getClass().getName());
        }
      };

      t.start();
      System.out.println(Thread.currentThread().getName() + ": Go to sleep for a second.");
      Thread.sleep(1000);

      System.out.println(Thread.currentThread().getName() + ": I am now well rested and will try to interrupt training procedure.");
      t.interrupt();

      t.join();
      if (interrupted.get() == 1) {
        System.out.println(Thread.currentThread().getName() + ": Joined the thread now continue with the next classifier.");
      } else {
        System.err.println(Thread.currentThread().getName() + ": Interrupt did not work correctly.");
      }
    }

  }

}
