/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    ParallelMultipleClassifiersCombiner.java
 *    Copyright (C) 2009-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

/**
 * Abstract utility class for handling settings common to meta classifiers that build an ensemble in
 * parallel using multiple classifiers.
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision$
 */
public abstract class ParallelMultipleClassifiersCombiner extends MultipleClassifiersCombiner {

  /** For serialization */
  private static final long serialVersionUID = 728109028953726626L;

  /** The number of threads to have executing at any one time */
  protected int m_numExecutionSlots = 1;

  /** Pool of threads to train models with */
  protected transient ThreadPoolExecutor m_executorPool;

  /** The number of classifiers completed so far */
  protected int m_completed;

  /**
   * The number of classifiers that experienced a failure of some sort during construction
   */
  protected int m_failed;

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {

    Vector<Option> newVector = new Vector<>(1);

    newVector.addElement(new Option("\tNumber of execution slots.\n" + "\t(default 1 - i.e. no parallelism)", "num-slots", 1, "-num-slots <num>"));

    newVector.addAll(Collections.list(super.listOptions()));

    return newVector.elements();
  }

  /**
   * Parses a given list of options. Valid options are:
   * <p>
   *
   * -Z num <br>
   * Set the number of execution slots to use (default 1 - i.e. no parallelism).
   * <p>
   *
   * Options after -- are passed to the designated classifier.
   * <p>
   *
   * @param options
   *          the list of options as an array of strings
   * @exception Exception
   *              if an option is not supported
   */
  @Override
  public void setOptions(final String[] options) throws Exception {

    String iterations = Utils.getOption("num-slots", options);
    if (iterations.length() != 0) {
      this.setNumExecutionSlots(Integer.parseInt(iterations));
    } else {
      this.setNumExecutionSlots(1);
    }

    super.setOptions(options);
  }

  /**
   * Gets the current settings of the classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {

    Vector<String> options = new Vector<>();

    options.add("-num-slots");
    options.add("" + this.getNumExecutionSlots());

    Collections.addAll(options, super.getOptions());

    return options.toArray(new String[0]);
  }

  /**
   * Set the number of execution slots (threads) to use for building the members of the ensemble.
   *
   * @param numSlots
   *          the number of slots to use.
   */
  public void setNumExecutionSlots(final int numSlots) {
    this.m_numExecutionSlots = numSlots;
  }

  /**
   * Get the number of execution slots (threads) to use for building the members of the ensemble.
   *
   * @return the number of slots to use
   */
  public int getNumExecutionSlots() {
    return this.m_numExecutionSlots;
  }

  /**
   * Returns the tip text for this property
   * 
   * @return tip text for this property suitable for displaying in the explorer/experimenter gui
   */
  public String numExecutionSlotsTipText() {
    return "The number of execution slots (threads) to use for " + "constructing the ensemble.";
  }

  /**
   * Stump method for building the classifiers
   *
   * @param data
   *          the training data to be used for generating the ensemble
   * @exception Exception
   *              if the classifier could not be built successfully
   */
  @Override
  public void buildClassifier(final Instances data) throws Exception {

    if (this.m_numExecutionSlots < 1) {
      throw new Exception("Number of execution slots needs to be >= 1!");
    }

    if (this.m_numExecutionSlots > 1) {
      if (this.m_Debug) {
        System.out.println("Starting executor pool with " + this.m_numExecutionSlots + " slots...");
      }
      this.startExecutorPool();
    }
    this.m_completed = 0;
    this.m_failed = 0;
  }

  /**
   * Start the pool of execution threads
   */
  protected void startExecutorPool() {
    if (this.m_executorPool != null) {
      this.m_executorPool.shutdownNow();
    }

    this.m_executorPool = new ThreadPoolExecutor(this.m_numExecutionSlots, this.m_numExecutionSlots, 120, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
  }

  private synchronized void block(final boolean tf) {
    if (tf) {
      try {
        if (this.m_numExecutionSlots > 1 && this.m_completed + this.m_failed < this.m_Classifiers.length) {
          this.wait();
        }
      } catch (InterruptedException ex) {
      }
    } else {
      this.notifyAll();
    }
  }

  /**
   * Does the actual construction of the ensemble
   *
   * @throws Exception
   *           if something goes wrong during the training process
   */
  protected synchronized void buildClassifiers(final Instances data) throws Exception {

    for (int i = 0; i < this.m_Classifiers.length; i++) {
      // XXX kill weka execution
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
      }
      if (this.m_numExecutionSlots > 1) {
        final Classifier currentClassifier = this.m_Classifiers[i];
        final int iteration = i;
        Runnable newTask = new Runnable() {
          @Override
          public void run() {
            try {
              if (ParallelMultipleClassifiersCombiner.this.m_Debug) {
                System.out.println("Training classifier (" + (iteration + 1) + ")");
              }
              currentClassifier.buildClassifier(data);
              if (ParallelMultipleClassifiersCombiner.this.m_Debug) {
                System.out.println("Finished classifier (" + (iteration + 1) + ")");
              }
              ParallelMultipleClassifiersCombiner.this.completedClassifier(iteration, true);
            } catch (Exception ex) {
              ex.printStackTrace();
              ParallelMultipleClassifiersCombiner.this.completedClassifier(iteration, false);
            }
          }
        };

        // launch this task
        this.m_executorPool.execute(newTask);
      } else {
        this.m_Classifiers[i].buildClassifier(data);
      }
    }

    if (this.m_numExecutionSlots > 1 && this.m_completed + this.m_failed < this.m_Classifiers.length) {
      this.block(true);
    }
  }

  /**
   * Records the completion of the training of a single classifier. Unblocks if all classifiers have
   * been trained.
   *
   * @param iteration
   *          the iteration that has completed
   * @param success
   *          whether the classifier trained successfully
   */
  protected synchronized void completedClassifier(final int iteration, final boolean success) {

    if (!success) {
      this.m_failed++;
      if (this.m_Debug) {
        System.err.println("Iteration " + iteration + " failed!");
      }
    } else {
      this.m_completed++;
    }

    if (this.m_completed + this.m_failed == this.m_Classifiers.length) {
      if (this.m_failed > 0) {
        if (this.m_Debug) {
          System.err.println("Problem building classifiers - some iterations failed.");
        }
      }

      // have to shut the pool down or program executes as a server
      // and when running from the command line does not return to the
      // prompt
      this.m_executorPool.shutdown();
      this.block(false);
    }
  }
}
