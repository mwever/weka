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
 *    GaussianProcesses.java
 *    Copyright (C) 2005-2012,2015 University of Waikato
 */

package weka.classifiers.functions;

import java.util.Collections;
import java.util.Enumeration;

import no.uib.cipr.matrix.DenseCholesky;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.UpperSPDDenseMatrix;
import no.uib.cipr.matrix.Vector;
import weka.classifiers.ConditionalDensityEstimator;
import weka.classifiers.IntervalEstimator;
import weka.classifiers.RandomizableClassifier;
import weka.classifiers.functions.supportVector.CachedKernel;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.Statistics;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * <!-- globalinfo-start --> * Implements Gaussian processes for regression without
 * hyperparameter-tuning. To make choosing an appropriate noise level easier, this implementation
 * applies normalization/standardization to the target attribute as well as the other attributes (if
 * normalization/standardizaton is turned on). Missing values are replaced by the global mean/mode.
 * Nominal attributes are converted to binary ones. Note that kernel caching is turned off if the
 * kernel used implements CachedKernel. * <br>
 * <br>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> * BibTeX: *
 *
 * <pre>
 * * &#64;misc{Mackay1998,
 * *    address = {Dept. of Physics, Cambridge University, UK},
 * *    author = {David J.C. Mackay},
 * *    title = {Introduction to Gaussian Processes},
 * *    year = {1998},
 * *    PS = {http://wol.ra.phy.cam.ac.uk/mackay/gpB.ps.gz}
 * * }
 * *
 * </pre>
 *
 * * <br>
 * <br>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> * Valid options are:
 * <p>
 * * *
 *
 * <pre>
 *  -L &lt;double&gt;
 * *  Level of Gaussian Noise wrt transformed target. (default 1)
 * </pre>
 *
 * * *
 *
 * <pre>
 *  -N
 * *  Whether to 0=normalize/1=standardize/2=neither. (default 0=normalize)
 * </pre>
 *
 * * *
 *
 * <pre>
 *  -K &lt;classname and parameters&gt;
 * *  The Kernel to use.
 * *  (default: weka.classifiers.functions.supportVector.PolyKernel)
 * </pre>
 *
 * * *
 *
 * <pre>
 *  -S &lt;num&gt;
 * *  Random number seed.
 * *  (default 1)
 * </pre>
 *
 * * *
 *
 * <pre>
 *  -output-debug-info
 * *  If set, classifier is run in debug mode and
 * *  may output additional info to the console
 * </pre>
 *
 * * *
 *
 * <pre>
 *  -do-not-check-capabilities
 * *  If set, classifier capabilities are not checked before classifier is built
 * *  (use with caution).
 * </pre>
 *
 * * *
 *
 * <pre>
 *  -num-decimal-places
 * *  The number of decimal places for the output of numbers in the model (default 2).
 * </pre>
 *
 * * *
 *
 * <pre>
 *
 * * Options specific to kernel weka.classifiers.functions.supportVector.PolyKernel:
 * *
 * </pre>
 *
 * * *
 *
 * <pre>
 *  -E &lt;num&gt;
 * *  The Exponent to use.
 * *  (default: 1.0)
 * </pre>
 *
 * * *
 *
 * <pre>
 *  -L
 * *  Use lower-order terms.
 * *  (default: no)
 * </pre>
 *
 * * *
 *
 * <pre>
 *  -C &lt;num&gt;
 * *  The size of the cache (a prime number), 0 for full cache and
 * *  -1 to turn it off.
 * *  (default: 250007)
 * </pre>
 *
 * * *
 *
 * <pre>
 *  -output-debug-info
 * *  Enables debugging output (if available) to be printed.
 * *  (default: off)
 * </pre>
 *
 * * *
 *
 * <pre>
 *  -no-checks
 * *  Turns off all checks - use with caution!
 * *  (default: checks on)
 * </pre>
 *
 * * <!-- options-end -->
 *
 * @author Kurt Driessens (kurtd@cs.waikato.ac.nz)
 * @author Remco Bouckaert (remco@cs.waikato.ac.nz)
 * @author Eibe Frank, University of Waikato
 * @version $Revision$
 */
public class GaussianProcesses extends RandomizableClassifier implements IntervalEstimator, ConditionalDensityEstimator, TechnicalInformationHandler, WeightedInstancesHandler {

  /** for serialization */
  static final long serialVersionUID = -8620066949967678545L;

  /** The filter used to make attributes numeric. */
  protected NominalToBinary m_NominalToBinary;

  /** normalizes the data */
  public static final int FILTER_NORMALIZE = 0;

  /** standardizes the data */
  public static final int FILTER_STANDARDIZE = 1;

  /** no filter */
  public static final int FILTER_NONE = 2;

  /** The filter to apply to the training data */
  public static final Tag[] TAGS_FILTER = { new Tag(FILTER_NORMALIZE, "Normalize training data"), new Tag(FILTER_STANDARDIZE, "Standardize training data"),
      new Tag(FILTER_NONE, "No normalization/standardization"), };

  /** The filter used to standardize/normalize all values. */
  protected Filter m_Filter = null;

  /** Whether to normalize/standardize/neither */
  protected int m_filterType = FILTER_NORMALIZE;

  /** The filter used to get rid of missing values. */
  protected ReplaceMissingValues m_Missing;

  /**
   * Turn off all checks and conversions? Turning them off assumes that data is purely numeric,
   * doesn't contain any missing values, and has a numeric class.
   */
  protected boolean m_checksTurnedOff = false;

  /** Gaussian Noise Value. */
  protected double m_delta = 1;

  /** The squared noise value. */
  protected double m_deltaSquared = 1;

  /**
   * The parameters of the linear transformation realized by the filter on the class attribute
   */
  protected double m_Alin;
  protected double m_Blin;

  /** Template of kernel to use */
  protected Kernel m_kernel = new PolyKernel();

  /** Actual kernel object to use */
  protected Kernel m_actualKernel;

  /** The number of training instances */
  protected int m_NumTrain = 0;

  /** The training data. */
  protected double m_avg_target;

  /** (negative) covariance matrix in symmetric matrix representation **/
  public Matrix m_L;

  /** The vector of target values. */
  protected Vector m_t;

  /** The weight of the training instances. */
  protected double[] m_weights;

  /**
   * Returns a string describing classifier
   *
   * @return a description suitable for displaying in the explorer/experimenter gui
   */
  public String globalInfo() {

    return " Implements Gaussian processes for " + "regression without hyperparameter-tuning. To make choosing an " + "appropriate noise level easier, this implementation applies "
        + "normalization/standardization to the target attribute as well " + "as the other attributes (if " + " normalization/standardizaton is turned on). Missing values "
        + "are replaced by the global mean/mode. Nominal attributes are " + "converted to binary ones. Note that kernel caching is turned off "
        + "if the kernel used implements CachedKernel.";
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing detailed information about the
   * technical background of this class, e.g., paper reference or book this class is based on.
   *
   * @return the technical information about this class
   */
  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;

    result = new TechnicalInformation(Type.MISC);
    result.setValue(Field.AUTHOR, "David J.C. Mackay");
    result.setValue(Field.YEAR, "1998");
    result.setValue(Field.TITLE, "Introduction to Gaussian Processes");
    result.setValue(Field.ADDRESS, "Dept. of Physics, Cambridge University, UK");
    result.setValue(Field.PS, "http://wol.ra.phy.cam.ac.uk/mackay/gpB.ps.gz");

    return result;
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = this.getKernel().getCapabilities();
    result.setOwner(this);

    // attribute
    result.enableAllAttributeDependencies();
    // with NominalToBinary we can also handle nominal attributes, but only
    // if the kernel can handle numeric attributes
    if (result.handles(Capability.NUMERIC_ATTRIBUTES)) {
      result.enable(Capability.NOMINAL_ATTRIBUTES);
    }
    result.enable(Capability.MISSING_VALUES);

    // class
    result.disableAllClasses();
    result.disableAllClassDependencies();
    result.disable(Capability.NO_CLASS);
    result.enable(Capability.NUMERIC_CLASS);
    result.enable(Capability.DATE_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    return result;
  }

  /**
   * Method for building the classifier.
   *
   * @param insts
   *          the set of training instances
   * @throws Exception
   *           if the classifier can't be built successfully
   */
  @Override
  public void buildClassifier(Instances insts) throws Exception {

    // check the set of training instances
    if (!this.m_checksTurnedOff) {
      // can classifier handle the data?
      this.getCapabilities().testWithFail(insts);

      // remove instances with missing class
      insts = new Instances(insts);
      insts.deleteWithMissingClass();
      this.m_Missing = new ReplaceMissingValues();
      this.m_Missing.setInputFormat(insts);
      insts = Filter.useFilter(insts, this.m_Missing);
    } else {
      this.m_Missing = null;
    }

    if (this.getCapabilities().handles(Capability.NUMERIC_ATTRIBUTES)) {
      boolean onlyNumeric = true;
      if (!this.m_checksTurnedOff) {
        for (int i = 0; i < insts.numAttributes(); i++) {
          // XXX kill weka execution
          if (Thread.currentThread().isInterrupted()) {
            throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
          }
          if (i != insts.classIndex()) {
            if (!insts.attribute(i).isNumeric()) {
              onlyNumeric = false;
              break;
            }
          }
        }
      }

      if (!onlyNumeric) {
        this.m_NominalToBinary = new NominalToBinary();
        this.m_NominalToBinary.setInputFormat(insts);
        insts = Filter.useFilter(insts, this.m_NominalToBinary);
      } else {
        this.m_NominalToBinary = null;
      }
    } else {
      this.m_NominalToBinary = null;
    }

    if (this.m_filterType == FILTER_STANDARDIZE) {
      this.m_Filter = new Standardize();
      ((Standardize) this.m_Filter).setIgnoreClass(true);
      this.m_Filter.setInputFormat(insts);
      insts = Filter.useFilter(insts, this.m_Filter);
    } else if (this.m_filterType == FILTER_NORMALIZE) {
      this.m_Filter = new Normalize();
      ((Normalize) this.m_Filter).setIgnoreClass(true);
      this.m_Filter.setInputFormat(insts);
      insts = Filter.useFilter(insts, this.m_Filter);
    } else {
      this.m_Filter = null;
    }

    this.m_NumTrain = insts.numInstances();

    // determine which linear transformation has been
    // applied to the class by the filter
    if (this.m_Filter != null) {
      Instance witness = (Instance) insts.instance(0).copy();
      witness.setValue(insts.classIndex(), 0);
      this.m_Filter.input(witness);
      this.m_Filter.batchFinished();
      Instance res = this.m_Filter.output();
      this.m_Blin = res.value(insts.classIndex());
      witness.setValue(insts.classIndex(), 1);
      this.m_Filter.input(witness);
      this.m_Filter.batchFinished();
      res = this.m_Filter.output();
      this.m_Alin = res.value(insts.classIndex()) - this.m_Blin;
    } else {
      this.m_Alin = 1.0;
      this.m_Blin = 0.0;
    }

    // Initialize kernel
    this.m_actualKernel = Kernel.makeCopy(this.m_kernel);
    if (this.m_kernel instanceof CachedKernel) {
      ((CachedKernel) this.m_actualKernel).setCacheSize(-1); // We don't need a cache at all
    }
    this.m_actualKernel.buildKernel(insts);

    // Compute average target value
    double sum = 0.0;
    for (int i = 0; i < insts.numInstances(); i++) {
      sum += insts.instance(i).weight() * insts.instance(i).classValue();
    }
    this.m_avg_target = sum / insts.sumOfWeights();

    // Store squared noise level
    this.m_deltaSquared = this.m_delta * this.m_delta;

    // Store square roots of instance m_weights
    this.m_weights = new double[insts.numInstances()];
    for (int i = 0; i < insts.numInstances(); i++) {
      this.m_weights[i] = Math.sqrt(insts.instance(i).weight());
    }

    // initialize kernel matrix/covariance matrix
    int n = insts.numInstances();
    // XXX kill weka execution
    if (Thread.currentThread().isInterrupted()) {
      throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
    }
    this.m_L = new UpperSPDDenseMatrix(n);
    for (int i = 0; i < n; i++) {
      // XXX kill weka execution
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
      }
      for (int j = i + 1; j < n; j++) {
        // XXX kill weka execution
        if (Thread.currentThread().isInterrupted()) {
          throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
        }
        this.m_L.set(i, j, this.m_weights[i] * this.m_weights[j] * this.m_actualKernel.eval(i, j, insts.instance(i)));
      }
      this.m_L.set(i, i, this.m_weights[i] * this.m_weights[i] * this.m_actualKernel.eval(i, i, insts.instance(i)) + this.m_deltaSquared);
    }

    // Compute inverse of kernel matrix
    this.m_L = new DenseCholesky(n, true).factor((UpperSPDDenseMatrix) this.m_L).solve(Matrices.identity(n));
    this.m_L = new UpperSPDDenseMatrix(this.m_L); // Convert from DenseMatrix

    // Compute t
    Vector tt = new DenseVector(n);
    for (int i = 0; i < n; i++) {
      // XXX kill weka execution
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
      }
      tt.set(i, this.m_weights[i] * (insts.instance(i).classValue() - this.m_avg_target));
    }
    this.m_t = this.m_L.mult(tt, new DenseVector(insts.numInstances()));

  } // buildClassifier

  /**
   * Classifies a given instance.
   *
   * @param inst
   *          the instance to be classified
   * @return the classification
   * @throws Exception
   *           if instance could not be classified successfully
   */
  @Override
  public double classifyInstance(Instance inst) throws Exception {

    // Filter instance
    inst = this.filterInstance(inst);

    // Build K vector
    Vector k = new DenseVector(this.m_NumTrain);
    for (int i = 0; i < this.m_NumTrain; i++) {
      // XXX kill weka execution
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
      }
      k.set(i, this.m_weights[i] * this.m_actualKernel.eval(-1, i, inst));
    }

    double result = (k.dot(this.m_t) + this.m_avg_target - this.m_Blin) / this.m_Alin;

    return result;

  }

  /**
   * Filters an instance.
   */
  protected Instance filterInstance(Instance inst) throws Exception {

    if (!this.m_checksTurnedOff) {
      this.m_Missing.input(inst);
      this.m_Missing.batchFinished();
      inst = this.m_Missing.output();
    }

    if (this.m_NominalToBinary != null) {
      this.m_NominalToBinary.input(inst);
      this.m_NominalToBinary.batchFinished();
      inst = this.m_NominalToBinary.output();
    }

    if (this.m_Filter != null) {
      this.m_Filter.input(inst);
      this.m_Filter.batchFinished();
      inst = this.m_Filter.output();
    }
    return inst;
  }

  /**
   * Computes standard deviation for given instance, without transforming target back into original
   * space.
   */
  protected double computeStdDev(final Instance inst, final Vector k) throws Exception {

    double kappa = this.m_actualKernel.eval(-1, -1, inst) + this.m_deltaSquared;

    double s = this.m_L.mult(k, new DenseVector(k.size())).dot(k);

    double sigma = this.m_delta;
    if (kappa > s) {
      sigma = Math.sqrt(kappa - s);
    }

    return sigma;
  }

  /**
   * Computes a prediction interval for the given instance and confidence level.
   *
   * @param inst
   *          the instance to make the prediction for
   * @param confidenceLevel
   *          the percentage of cases the interval should cover
   * @return a 1*2 array that contains the boundaries of the interval
   * @throws Exception
   *           if interval could not be estimated successfully
   */
  @Override
  public double[][] predictIntervals(Instance inst, double confidenceLevel) throws Exception {

    inst = this.filterInstance(inst);

    // Build K vector (and Kappa)
    Vector k = new DenseVector(this.m_NumTrain);
    for (int i = 0; i < this.m_NumTrain; i++) {
      k.set(i, this.m_weights[i] * this.m_actualKernel.eval(-1, i, inst));
    }

    double estimate = k.dot(this.m_t) + this.m_avg_target;

    double sigma = this.computeStdDev(inst, k);

    confidenceLevel = 1.0 - ((1.0 - confidenceLevel) / 2.0);

    double z = Statistics.normalInverse(confidenceLevel);

    double[][] interval = new double[1][2];

    interval[0][0] = estimate - z * sigma;
    interval[0][1] = estimate + z * sigma;

    interval[0][0] = (interval[0][0] - this.m_Blin) / this.m_Alin;
    interval[0][1] = (interval[0][1] - this.m_Blin) / this.m_Alin;

    return interval;

  }

  /**
   * Gives standard deviation of the prediction at the given instance.
   *
   * @param inst
   *          the instance to get the standard deviation for
   * @return the standard deviation
   * @throws Exception
   *           if computation fails
   */
  public double getStandardDeviation(Instance inst) throws Exception {

    inst = this.filterInstance(inst);

    // Build K vector (and Kappa)
    Vector k = new DenseVector(this.m_NumTrain);
    for (int i = 0; i < this.m_NumTrain; i++) {
      // XXX kill weka execution
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
      }
      k.set(i, this.m_weights[i] * this.m_actualKernel.eval(-1, i, inst));
    }

    return this.computeStdDev(inst, k) / this.m_Alin;
  }

  /**
   * Returns natural logarithm of density estimate for given value based on given instance.
   *
   * @param inst
   *          the instance to make the prediction for.
   * @param value
   *          the value to make the prediction for.
   * @return the natural logarithm of the density estimate
   * @exception Exception
   *              if the density cannot be computed
   */
  @Override
  public double logDensity(Instance inst, double value) throws Exception {

    inst = this.filterInstance(inst);

    // Build K vector (and Kappa)
    Vector k = new DenseVector(this.m_NumTrain);
    for (int i = 0; i < this.m_NumTrain; i++) {
      k.set(i, this.m_weights[i] * this.m_actualKernel.eval(-1, i, inst));
    }

    double estimate = k.dot(this.m_t) + this.m_avg_target;

    double sigma = this.computeStdDev(inst, k);

    // transform to GP space
    value = value * this.m_Alin + this.m_Blin;
    // center around estimate
    value = value - estimate;
    double z = -Math.log(sigma * Math.sqrt(2 * Math.PI)) - value * value / (2.0 * sigma * sigma);

    return z + Math.log(this.m_Alin);
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {

    java.util.Vector<Option> result = new java.util.Vector<>();

    result.addElement(new Option("\tLevel of Gaussian Noise wrt transformed target." + " (default 1)", "L", 1, "-L <double>"));

    result.addElement(new Option("\tWhether to 0=normalize/1=standardize/2=neither. " + "(default 0=normalize)", "N", 1, "-N"));

    result.addElement(new Option("\tThe Kernel to use.\n" + "\t(default: weka.classifiers.functions.supportVector.PolyKernel)", "K", 1, "-K <classname and parameters>"));

    result.addAll(Collections.list(super.listOptions()));

    result.addElement(new Option("", "", 0, "\nOptions specific to kernel " + this.getKernel().getClass().getName() + ":"));

    result.addAll(Collections.list(((OptionHandler) this.getKernel()).listOptions()));

    return result.elements();
  }

  /**
   * Parses a given list of options.
   * <p/>
   *
   * <!-- options-start --> * Valid options are:
   * <p>
   * * *
   *
   * <pre>
   *  -L &lt;double&gt;
   * *  Level of Gaussian Noise wrt transformed target. (default 1)
   * </pre>
   *
   * * *
   *
   * <pre>
   *  -N
   * *  Whether to 0=normalize/1=standardize/2=neither. (default 0=normalize)
   * </pre>
   *
   * * *
   *
   * <pre>
   *  -K &lt;classname and parameters&gt;
   * *  The Kernel to use.
   * *  (default: weka.classifiers.functions.supportVector.PolyKernel)
   * </pre>
   *
   * * *
   *
   * <pre>
   *  -S &lt;num&gt;
   * *  Random number seed.
   * *  (default 1)
   * </pre>
   *
   * * *
   *
   * <pre>
   *  -output-debug-info
   * *  If set, classifier is run in debug mode and
   * *  may output additional info to the console
   * </pre>
   *
   * * *
   *
   * <pre>
   *  -do-not-check-capabilities
   * *  If set, classifier capabilities are not checked before classifier is built
   * *  (use with caution).
   * </pre>
   *
   * * *
   *
   * <pre>
   *  -num-decimal-places
   * *  The number of decimal places for the output of numbers in the model (default 2).
   * </pre>
   *
   * * *
   *
   * <pre>
   *
   * * Options specific to kernel weka.classifiers.functions.supportVector.PolyKernel:
   * *
   * </pre>
   *
   * * *
   *
   * <pre>
   *  -E &lt;num&gt;
   * *  The Exponent to use.
   * *  (default: 1.0)
   * </pre>
   *
   * * *
   *
   * <pre>
   *  -L
   * *  Use lower-order terms.
   * *  (default: no)
   * </pre>
   *
   * * *
   *
   * <pre>
   *  -C &lt;num&gt;
   * *  The size of the cache (a prime number), 0 for full cache and
   * *  -1 to turn it off.
   * *  (default: 250007)
   * </pre>
   *
   * * *
   *
   * <pre>
   *  -output-debug-info
   * *  Enables debugging output (if available) to be printed.
   * *  (default: off)
   * </pre>
   *
   * * *
   *
   * <pre>
   *  -no-checks
   * *  Turns off all checks - use with caution!
   * *  (default: checks on)
   * </pre>
   *
   * * <!-- options-end -->
   *
   * @param options
   *          the list of options as an array of strings
   * @throws Exception
   *           if an option is not supported
   */
  @Override
  public void setOptions(final String[] options) throws Exception {
    String tmpStr;
    String[] tmpOptions;

    tmpStr = Utils.getOption('L', options);
    if (tmpStr.length() != 0) {
      this.setNoise(Double.parseDouble(tmpStr));
    } else {
      this.setNoise(1);
    }

    tmpStr = Utils.getOption('N', options);
    if (tmpStr.length() != 0) {
      this.setFilterType(new SelectedTag(Integer.parseInt(tmpStr), TAGS_FILTER));
    } else {
      this.setFilterType(new SelectedTag(FILTER_NORMALIZE, TAGS_FILTER));
    }

    tmpStr = Utils.getOption('K', options);
    tmpOptions = Utils.splitOptions(tmpStr);
    if (tmpOptions.length != 0) {
      tmpStr = tmpOptions[0];
      tmpOptions[0] = "";
      this.setKernel(Kernel.forName(tmpStr, tmpOptions));
    }

    super.setOptions(options);

    Utils.checkForRemainingOptions(options);
  }

  /**
   * Gets the current settings of the classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {

    java.util.Vector<String> result = new java.util.Vector<>();

    result.addElement("-L");
    result.addElement("" + this.getNoise());

    result.addElement("-N");
    result.addElement("" + this.m_filterType);

    result.addElement("-K");
    result.addElement("" + this.m_kernel.getClass().getName() + " " + Utils.joinOptions(this.m_kernel.getOptions()));

    Collections.addAll(result, super.getOptions());

    return result.toArray(new String[result.size()]);
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the explorer/experimenter gui
   */
  public String kernelTipText() {
    return "The kernel to use.";
  }

  /**
   * Gets the kernel to use.
   *
   * @return the kernel
   */
  public Kernel getKernel() {
    return this.m_kernel;
  }

  /**
   * Sets the kernel to use.
   *
   * @param value
   *          the new kernel
   */
  public void setKernel(final Kernel value) {
    this.m_kernel = value;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the explorer/experimenter gui
   */
  public String filterTypeTipText() {
    return "Determines how/if the data will be transformed.";
  }

  /**
   * Gets how the training data will be transformed. Will be one of FILTER_NORMALIZE,
   * FILTER_STANDARDIZE, FILTER_NONE.
   *
   * @return the filtering mode
   */
  public SelectedTag getFilterType() {

    return new SelectedTag(this.m_filterType, TAGS_FILTER);
  }

  /**
   * Sets how the training data will be transformed. Should be one of FILTER_NORMALIZE,
   * FILTER_STANDARDIZE, FILTER_NONE.
   *
   * @param newType
   *          the new filtering mode
   */
  public void setFilterType(final SelectedTag newType) {

    if (newType.getTags() == TAGS_FILTER) {
      this.m_filterType = newType.getSelectedTag().getID();
    }
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the explorer/experimenter gui
   */
  public String noiseTipText() {
    return "The level of Gaussian Noise (added to the diagonal of the Covariance Matrix), after the " + "target has been normalized/standardized/left unchanged).";
  }

  /**
   * Get the value of noise.
   *
   * @return Value of noise.
   */
  public double getNoise() {
    return this.m_delta;
  }

  /**
   * Set the level of Gaussian Noise.
   *
   * @param v
   *          Value to assign to noise.
   */
  public void setNoise(final double v) {
    this.m_delta = v;
  }

  /**
   * Prints out the classifier.
   *
   * @return a description of the classifier as a string
   */
  @Override
  public String toString() {

    StringBuffer text = new StringBuffer();

    if (this.m_t == null) {
      return "Gaussian Processes: No model built yet.";
    }

    try {

      text.append("Gaussian Processes\n\n");
      text.append("Kernel used:\n  " + this.m_kernel.toString() + "\n\n");

      text.append("All values shown based on: " + TAGS_FILTER[this.m_filterType].getReadable() + "\n\n");

      text.append("Average Target Value : " + this.m_avg_target + "\n");

      text.append("Inverted Covariance Matrix:\n");
      double min = this.m_L.get(0, 0);
      double max = this.m_L.get(0, 0);
      for (int i = 0; i < this.m_NumTrain; i++) {
        for (int j = 0; j <= i; j++) {
          if (this.m_L.get(i, j) < min) {
            min = this.m_L.get(i, j);
          } else if (this.m_L.get(i, j) > max) {
            max = this.m_L.get(i, j);
          }
        }
      }
      text.append("    Lowest Value = " + min + "\n");
      text.append("    Highest Value = " + max + "\n");
      text.append("Inverted Covariance Matrix * Target-value Vector:\n");
      min = this.m_t.get(0);
      max = this.m_t.get(0);
      for (int i = 0; i < this.m_NumTrain; i++) {
        if (this.m_t.get(i) < min) {
          min = this.m_t.get(i);
        } else if (this.m_t.get(i) > max) {
          max = this.m_t.get(i);
        }
      }
      text.append("    Lowest Value = " + min + "\n");
      text.append("    Highest Value = " + max + "\n \n");

    } catch (Exception e) {
      return "Can't print the classifier.";
    }

    return text.toString();
  }

  /**
   * Main method for testing this class.
   *
   * @param argv
   *          the commandline parameters
   */
  public static void main(final String[] argv) {

    runClassifier(new GaussianProcesses(), argv);
  }
}
