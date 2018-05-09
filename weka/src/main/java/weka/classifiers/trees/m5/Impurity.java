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
 *    Impurity.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.m5;

import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

/**
 * Class for handling the impurity values when spliting the instances
 *
 * @author Yong Wang (yongwang@cs.waikato.ac.nz)
 * @version $Revision$
 */
public final class Impurity implements RevisionHandler {

  double n; // number of total instances
  int attr; // splitting attribute
  double nl; // number of instances in the left group
  double nr; // number of instances in the right group
  double sl; // sum of the left group
  double sr; // sum of the right group
  double s2l; // squared sum of the left group
  double s2r; // squared sum of the right group
  double sdl; // standard deviation of the left group
  double sdr; // standard deviation of the right group
  double vl; // variance of the left group
  double vr; // variance of the right group
  double sd; // overall standard deviation
  double va; // overall variance

  double impurity; // impurity value;
  int order; // order = 1, variance; order = 2, standard deviation; order = 3, the cubic root of the variance;
             // order = k, the k-th order root of the variance

  /**
   * Constructs an Impurity object containing the impurity values of partitioning the instances using
   * an attribute
   *
   * @param partition
   *          the index of the last instance in the left subset
   * @param attribute
   *          the attribute used in partitioning
   * @param inst
   *          instances
   * @param k
   *          the order of the impurity; =1, the variance; =2, the stardard deviation; =k, the k-th
   *          order root of the variance
   * @throws InterruptedException
   */
  public Impurity(final int partition, final int attribute, final Instances inst, final int k) throws InterruptedException {

    Values values = new Values(0, inst.numInstances() - 1, inst.classIndex(), inst);
    this.attr = attribute;
    this.n = inst.numInstances();
    this.sd = values.sd;
    this.va = values.va;

    values = new Values(0, partition, inst.classIndex(), inst);
    this.nl = partition + 1;
    this.sl = values.sum;
    this.s2l = values.sqrSum;

    values = new Values(partition + 1, inst.numInstances() - 1, inst.classIndex(), inst);
    this.nr = inst.numInstances() - partition - 1;
    this.sr = values.sum;
    this.s2r = values.sqrSum;

    this.order = k;
    this.incremental(0, 0);
  }

  /**
   * Converts an Impurity object to a string
   *
   * @return the converted string
   */
  @Override
  public final String toString() {

    StringBuffer text = new StringBuffer();

    text.append("Print impurity values:\n");
    text.append("    Number of total instances:\t" + this.n + "\n");
    text.append("    Splitting attribute:\t\t" + this.attr + "\n");
    text.append("    Number of the instances in the left:\t" + this.nl + "\n");
    text.append("    Number of the instances in the right:\t" + this.nr + "\n");
    text.append("    Sum of the left:\t\t\t" + this.sl + "\n");
    text.append("    Sum of the right:\t\t\t" + this.sr + "\n");
    text.append("    Squared sum of the left:\t\t" + this.s2l + "\n");
    text.append("    Squared sum of the right:\t\t" + this.s2r + "\n");
    text.append("    Standard deviation of the left:\t" + this.sdl + "\n");
    text.append("    Standard deviation of the right:\t" + this.sdr + "\n");
    text.append("    Variance of the left:\t\t" + this.vr + "\n");
    text.append("    Variance of the right:\t\t" + this.vr + "\n");
    text.append("    Overall standard deviation:\t\t" + this.sd + "\n");
    text.append("    Overall variance:\t\t\t" + this.va + "\n");
    text.append("    Impurity (order " + this.order + "):\t\t" + this.impurity + "\n");

    return text.toString();
  }

  /**
   * Incrementally computes the impurirty values
   *
   * @param value
   *          the incremental value
   * @param type
   *          if type=1, value will be added to the left subset; type=-1, to the right subset; type=0,
   *          initializes
   * @throws InterruptedException
   */
  public final void incremental(final double value, final int type) throws InterruptedException {
    // XXX kill weka execution
    if (Thread.currentThread().isInterrupted()) {
      throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
    }
    double y = 0., yl = 0., yr = 0.;

    switch (type) {
      case 1:
        this.nl += 1;
        this.nr -= 1;
        this.sl += value;
        this.sr -= value;
        this.s2l += value * value;
        this.s2r -= value * value;
        break;
      case -1:
        this.nl -= 1;
        this.nr += 1;
        this.sl -= value;
        this.sr += value;
        this.s2l -= value * value;
        this.s2r += value * value;
        break;
      case 0:
        break;
      default:
        System.err.println("wrong type in Impurity.incremental().");
    }

    if (this.nl <= 0.0) {
      this.vl = 0.0;
      this.sdl = 0.0;
    } else {
      this.vl = (this.nl * this.s2l - this.sl * this.sl) / (this.nl * (this.nl));
      this.vl = Math.abs(this.vl);
      this.sdl = Math.sqrt(this.vl);
    }
    if (this.nr <= 0.0) {
      this.vr = 0.0;
      this.sdr = 0.0;
    } else {
      this.vr = (this.nr * this.s2r - this.sr * this.sr) / (this.nr * (this.nr));
      this.vr = Math.abs(this.vr);
      this.sdr = Math.sqrt(this.vr);
    }

    if (this.order <= 0) {
      System.err.println("Impurity order less than zero in Impurity.incremental()");
    } else if (this.order == 1) {
      y = this.va;
      yl = this.vl;
      yr = this.vr;
    } else {
      y = Math.pow(this.va, 1. / this.order);
      yl = Math.pow(this.vl, 1. / this.order);
      yr = Math.pow(this.vr, 1. / this.order);
    }

    if (this.nl <= 0.0 || this.nr <= 0.0) {
      this.impurity = 0.0;
    } else {
      this.impurity = y - (this.nl / this.n) * yl - (this.nr / this.n) * yr;
    }
  }

  /**
   * Returns the revision string.
   *
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision$");
  }
}
