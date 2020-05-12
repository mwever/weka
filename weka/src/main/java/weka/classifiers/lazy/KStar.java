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
 *    KStar.java
 *    Copyright (C) 1995-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.lazy;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.lazy.kstar.KStarCache;
import weka.classifiers.lazy.kstar.KStarConstants;
import weka.classifiers.lazy.kstar.KStarNominalAttribute;
import weka.classifiers.lazy.kstar.KStarNumericAttribute;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

/**
 * <!-- globalinfo-start --> K* is an instance-based classifier, that is the class of a test instance is based upon the class of those training instances similar to it, as determined by some similarity function. It differs from other
 * instance-based learners in that it uses an entropy-based distance function.<br/>
 * <br/>
 * For more information on K*, see<br/>
 * <br/>
 * John G. Cleary, Leonard E. Trigg: K*: An Instance-based Learner Using an Entropic Distance Measure. In: 12th International Conference on Machine Learning, 108-114, 1995.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#64;inproceedings{Cleary1995,
 *    author = {John G. Cleary and Leonard E. Trigg},
 *    booktitle = {12th International Conference on Machine Learning},
 *    pages = {108-114},
 *    title = {K*: An Instance-based Learner Using an Entropic Distance Measure},
 *    year = {1995}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 *  -B &lt;num&gt;
 *  Manual blend setting (default 20%)
 * </pre>
 *
 * <pre>
 *  -E
 *  Enable entropic auto-blend setting (symbolic class only)
 * </pre>
 *
 * <pre>
 *  -M &lt;char&gt;
 *  Specify the missing value treatment mode (default a)
 *  Valid options are: a(verage), d(elete), m(axdiff), n(ormal)
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Len Trigg (len@reeltwo.com)
 * @author Abdelaziz Mahoui (am14@cs.waikato.ac.nz) - Java port
 * @version $Revision$
 */
public class KStar extends AbstractClassifier implements KStarConstants, UpdateableClassifier, TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = 332458330800479083L;

	/** The training instances used for classification. */
	protected Instances m_Train;

	/** The number of instances in the dataset */
	protected int m_NumInstances;

	/** The number of class values */
	protected int m_NumClasses;

	/** The number of attributes */
	protected int m_NumAttributes;

	/** The class attribute type */
	protected int m_ClassType;

	/** Table of random class value colomns */
	protected int[][] m_RandClassCols;

	/** Flag turning on and off the computation of random class colomns */
	protected int m_ComputeRandomCols = ON;

	/** Flag turning on and off the initialisation of config variables */
	protected int m_InitFlag = ON;

	/**
	 * A custom data structure for caching distinct attribute values and their scale factor or stop parameter.
	 */
	protected KStarCache[] m_Cache;

	/** missing value treatment */
	protected int m_MissingMode = M_AVERAGE;

	/** 0 = use specified blend, 1 = entropic blend setting */
	protected int m_BlendMethod = B_SPHERE;

	/** default sphere of influence blend setting */
	protected int m_GlobalBlend = 20;

	/** Define possible missing value handling methods */
	public static final Tag[] TAGS_MISSING = { new Tag(M_DELETE, "Ignore the instances with missing values"), new Tag(M_MAXDIFF, "Treat missing values as maximally different"), new Tag(M_NORMAL, "Normalize over the attributes"),
			new Tag(M_AVERAGE, "Average column entropy curves") };

	/**
	 * Returns a string describing classifier
	 *
	 * @return a description suitable for displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return "K* is an instance-based classifier, that is the class of a test " + "instance is based upon the class of those training instances " + "similar to it, as determined by some similarity function.  It differs "
				+ "from other instance-based learners in that it uses an entropy-based " + "distance function.\n\n" + "For more information on K*, see\n\n" + this.getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed information about the technical background of this class, e.g., paper reference or book this class is based on.
	 *
	 * @return the technical information about this class
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.INPROCEEDINGS);
		result.setValue(Field.AUTHOR, "John G. Cleary and Leonard E. Trigg");
		result.setValue(Field.TITLE, "K*: An Instance-based Learner Using an Entropic Distance Measure");
		result.setValue(Field.BOOKTITLE, "12th International Conference on Machine Learning");
		result.setValue(Field.YEAR, "1995");
		result.setValue(Field.PAGES, "108-114");

		return result;
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.DATE_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	/**
	 * Generates the classifier.
	 *
	 * @param instances
	 *            set of instances serving as training data
	 * @throws Exception
	 *             if the classifier has not been generated successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {

		// can classifier handle the data?
		this.getCapabilities().testWithFail(instances);

		// remove instances with missing class
		instances = new Instances(instances);
		instances.deleteWithMissingClass();

		this.m_Train = new Instances(instances, 0, instances.numInstances());

		// initializes class attributes ** java-speaking! :-) **
		this.init_m_Attributes();
	}

	/**
	 * Adds the supplied instance to the training set
	 *
	 * @param instance
	 *            the instance to add
	 * @throws Exception
	 *             if instance could not be incorporated successfully
	 */
	@Override
	public void updateClassifier(final Instance instance) throws Exception {

		if (this.m_Train.equalHeaders(instance.dataset()) == false) {
			throw new Exception("Incompatible instance types\n" + this.m_Train.equalHeadersMsg(instance.dataset()));
		}
		if (instance.classIsMissing()) {
			return;
		}
		this.m_Train.add(instance);
		// update relevant attributes ...
		this.update_m_Attributes();
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param instance
	 *            the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception
	 *             if an error occurred during the prediction
	 */
	@Override
	public double[] distributionForInstance(final Instance instance) throws Exception {

		double transProb = 0.0, temp = 0.0;
		double[] classProbability = new double[this.m_NumClasses];
		double[] predictedValue = new double[1];

		// initialization ...
		for (int i = 0; i < classProbability.length; i++) {
			classProbability[i] = 0.0;
		}
		predictedValue[0] = 0.0;
		if (this.m_InitFlag == ON) {
			// need to compute them only once and will be used for all instances.
			// We are doing this because the evaluation module controls the calls.
			if (this.m_BlendMethod == B_ENTROPY) {
				this.generateRandomClassColomns();
			}
			this.m_Cache = new KStarCache[this.m_NumAttributes];
			for (int i = 0; i < this.m_NumAttributes; i++) {
				this.m_Cache[i] = new KStarCache();
			}
			this.m_InitFlag = OFF;
		}
		// init done.
		Instance trainInstance;
		Enumeration<Instance> enu = this.m_Train.enumerateInstances();
		while (enu.hasMoreElements()) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			trainInstance = enu.nextElement();
			transProb = this.instanceTransformationProbability(instance, trainInstance);
			switch (this.m_ClassType) {
			case Attribute.NOMINAL:
				classProbability[(int) trainInstance.classValue()] += transProb;
				break;
			case Attribute.NUMERIC:
				predictedValue[0] += transProb * trainInstance.classValue();
				temp += transProb;
				break;
			}
		}
		if (this.m_ClassType == Attribute.NOMINAL) {
			double sum = Utils.sum(classProbability);
			if (sum <= 0.0) {
				for (int i = 0; i < classProbability.length; i++) {
					classProbability[i] = (double) 1 / (double) this.m_NumClasses;
				}
			} else {
				Utils.normalize(classProbability, sum);
			}
			return classProbability;
		} else {
			predictedValue[0] = (temp != 0) ? predictedValue[0] / temp : 0.0;
			return predictedValue;
		}
	}

	/**
	 * Calculate the probability of the first instance transforming into the second instance: the probability is the product of the transformation probabilities of the attributes normilized over the number of instances used.
	 *
	 * @param first
	 *            the test instance
	 * @param second
	 *            the train instance
	 * @return transformation probability value
	 * @throws Exception
	 */
	private double instanceTransformationProbability(final Instance first, final Instance second) throws Exception {
		double transProb = 1.0;
		int numMissAttr = 0;
		for (int i = 0; i < this.m_NumAttributes; i++) {
			// XXX interrupt weka
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			if (i == this.m_Train.classIndex()) {
				continue; // ignore class attribute
			}
			if (first.isMissing(i)) { // test instance attribute value is missing
				numMissAttr++;
				continue;
			}
			transProb *= this.attrTransProb(first, second, i);
			// normilize for missing values
			if (numMissAttr != this.m_NumAttributes) {
				transProb = Math.pow(transProb, (double) this.m_NumAttributes / (this.m_NumAttributes - numMissAttr));
			} else { // weird case!
				transProb = 0.0;
			}
		}
		// normilize for the train dataset
		return transProb / this.m_NumInstances;
	}

	/**
	 * Calculates the transformation probability of the indexed test attribute to the indexed train attribute.
	 *
	 * @param first
	 *            the test instance.
	 * @param second
	 *            the train instance.
	 * @param col
	 *            the index of the attribute in the instance.
	 * @return the value of the transformation probability.
	 * @throws Exception
	 */
	private double attrTransProb(final Instance first, final Instance second, final int col) throws Exception {

		double transProb = 0.0;
		KStarNominalAttribute ksNominalAttr;
		KStarNumericAttribute ksNumericAttr;
		switch (this.m_Train.attribute(col).type()) {
		case Attribute.NOMINAL:
			ksNominalAttr = new KStarNominalAttribute(first, second, col, this.m_Train, this.m_RandClassCols, this.m_Cache[col]);
			ksNominalAttr.setOptions(this.m_MissingMode, this.m_BlendMethod, this.m_GlobalBlend);
			transProb = ksNominalAttr.transProb();
			ksNominalAttr = null;
			break;

		case Attribute.NUMERIC:
			ksNumericAttr = new KStarNumericAttribute(first, second, col, this.m_Train, this.m_RandClassCols, this.m_Cache[col]);
			ksNumericAttr.setOptions(this.m_MissingMode, this.m_BlendMethod, this.m_GlobalBlend);
			transProb = ksNumericAttr.transProb();
			ksNumericAttr = null;
			break;
		}
		return transProb;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String missingModeTipText() {
		return "Determines how missing attribute values are treated.";
	}

	/**
	 * Gets the method to use for handling missing values. Will be one of M_NORMAL, M_AVERAGE, M_MAXDIFF or M_DELETE.
	 *
	 * @return the method used for handling missing values.
	 */
	public SelectedTag getMissingMode() {

		return new SelectedTag(this.m_MissingMode, TAGS_MISSING);
	}

	/**
	 * Sets the method to use for handling missing values. Values other than M_NORMAL, M_AVERAGE, M_MAXDIFF and M_DELETE will be ignored.
	 *
	 * @param newMode
	 *            the method to use for handling missing values.
	 */
	public void setMissingMode(final SelectedTag newMode) {

		if (newMode.getTags() == TAGS_MISSING) {
			this.m_MissingMode = newMode.getSelectedTag().getID();
		}
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> optVector = new Vector<>(3);
		optVector.addElement(new Option("\tManual blend setting (default 20%)\n", "B", 1, "-B <num>"));
		optVector.addElement(new Option("\tEnable entropic auto-blend setting (symbolic class only)\n", "E", 0, "-E"));
		optVector.addElement(new Option("\tSpecify the missing value treatment mode (default a)\n" + "\tValid options are: a(verage), d(elete), m(axdiff), n(ormal)\n", "M", 1, "-M <char>"));

		optVector.addAll(Collections.list(super.listOptions()));

		return optVector.elements();
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String globalBlendTipText() {
		return "The parameter for global blending. Values are restricted to [0,100].";
	}

	/**
	 * Set the global blend parameter
	 *
	 * @param b
	 *            the value for global blending
	 */
	public void setGlobalBlend(final int b) {
		this.m_GlobalBlend = b;
		if (this.m_GlobalBlend > 100) {
			this.m_GlobalBlend = 100;
		}
		if (this.m_GlobalBlend < 0) {
			this.m_GlobalBlend = 0;
		}
	}

	/**
	 * Get the value of the global blend parameter
	 *
	 * @return the value of the global blend parameter
	 */
	public int getGlobalBlend() {
		return this.m_GlobalBlend;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String entropicAutoBlendTipText() {
		return "Whether entropy-based blending is to be used.";
	}

	/**
	 * Set whether entropic blending is to be used.
	 *
	 * @param e
	 *            true if entropic blending is to be used
	 */
	public void setEntropicAutoBlend(final boolean e) {
		if (e) {
			this.m_BlendMethod = B_ENTROPY;
		} else {
			this.m_BlendMethod = B_SPHERE;
		}
	}

	/**
	 * Get whether entropic blending being used
	 *
	 * @return true if entropic blending is used
	 */
	public boolean getEntropicAutoBlend() {
		if (this.m_BlendMethod == B_ENTROPY) {
			return true;
		}

		return false;
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 *
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 *
	 * <pre>
	 *  -B &lt;num&gt;
	 *  Manual blend setting (default 20%)
	 * </pre>
	 *
	 * <pre>
	 *  -E
	 *  Enable entropic auto-blend setting (symbolic class only)
	 * </pre>
	 *
	 * <pre>
	 *  -M &lt;char&gt;
	 *  Specify the missing value treatment mode (default a)
	 *  Valid options are: a(verage), d(elete), m(axdiff), n(ormal)
	 * </pre>
	 *
	 * <!-- options-end -->
	 *
	 * @param options
	 *            the list of options as an array of strings
	 * @throws Exception
	 *             if an option is not supported
	 */
	@Override
	public void setOptions(final String[] options) throws Exception {

		String blendStr = Utils.getOption('B', options);
		if (blendStr.length() != 0) {
			this.setGlobalBlend(Integer.parseInt(blendStr));
		}

		this.setEntropicAutoBlend(Utils.getFlag('E', options));

		String missingModeStr = Utils.getOption('M', options);
		if (missingModeStr.length() != 0) {
			switch (missingModeStr.charAt(0)) {
			case 'a':
				this.setMissingMode(new SelectedTag(M_AVERAGE, TAGS_MISSING));
				break;
			case 'd':
				this.setMissingMode(new SelectedTag(M_DELETE, TAGS_MISSING));
				break;
			case 'm':
				this.setMissingMode(new SelectedTag(M_MAXDIFF, TAGS_MISSING));
				break;
			case 'n':
				this.setMissingMode(new SelectedTag(M_NORMAL, TAGS_MISSING));
				break;
			default:
				this.setMissingMode(new SelectedTag(M_AVERAGE, TAGS_MISSING));
			}
		}

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of K*.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	@Override
	public String[] getOptions() {
		// -B <num> -E -M <char>
		Vector<String> options = new Vector<>();

		options.add("-B");
		options.add("" + this.m_GlobalBlend);

		if (this.getEntropicAutoBlend()) {
			options.add("-E");
		}

		options.add("-M");
		if (this.m_MissingMode == M_AVERAGE) {
			options.add("" + "a");
		} else if (this.m_MissingMode == M_DELETE) {
			options.add("" + "d");
		} else if (this.m_MissingMode == M_MAXDIFF) {
			options.add("" + "m");
		} else if (this.m_MissingMode == M_NORMAL) {
			options.add("" + "n");
		}

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	}

	/**
	 * Returns a description of this classifier.
	 *
	 * @return a description of this classifier as a string.
	 */
	@Override
	public String toString() {
		StringBuffer st = new StringBuffer();
		st.append("KStar Beta Verion (0.1b).\n" + "Copyright (c) 1995-97 by Len Trigg (trigg@cs.waikato.ac.nz).\n" + "Java port to Weka by Abdelaziz Mahoui " + "(am14@cs.waikato.ac.nz).\n\nKStar options : ");
		String[] ops = this.getOptions();
		for (int i = 0; i < ops.length; i++) {
			st.append(ops[i] + ' ');
		}
		return st.toString();
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv
	 *            should contain command line options (see setOptions)
	 */
	public static void main(final String[] argv) {
		runClassifier(new KStar(), argv);
	}

	/**
	 * Initializes the m_Attributes of the class.
	 */
	private void init_m_Attributes() {
		try {
			this.m_NumInstances = this.m_Train.numInstances();
			this.m_NumClasses = this.m_Train.numClasses();
			this.m_NumAttributes = this.m_Train.numAttributes();
			this.m_ClassType = this.m_Train.classAttribute().type();
			this.m_InitFlag = ON;
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Updates the m_attributes of the class.
	 */
	private void update_m_Attributes() {
		this.m_NumInstances = this.m_Train.numInstances();
		this.m_InitFlag = ON;
	}

	/**
	 * Note: for Nominal Class Only! Generates a set of random versions of the class colomn.
	 */
	private void generateRandomClassColomns() {

		Random generator = new Random(42);
		// Random generator = new Random();
		this.m_RandClassCols = new int[NUM_RAND_COLS + 1][];
		int[] classvals = this.classValues();
		for (int i = 0; i < NUM_RAND_COLS; i++) {
			// generate a randomized version of the class colomn
			this.m_RandClassCols[i] = this.randomize(classvals, generator);
		}
		// original colomn is preserved in colomn NUM_RAND_COLS
		this.m_RandClassCols[NUM_RAND_COLS] = classvals;
	}

	/**
	 * Note: for Nominal Class Only! Returns an array of the class values
	 *
	 * @return an array of class values
	 */
	private int[] classValues() {
		int[] classval = new int[this.m_NumInstances];
		for (int i = 0; i < this.m_NumInstances; i++) {
			try {
				classval[i] = (int) this.m_Train.instance(i).classValue();
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}
		return classval;
	}

	/**
	 * Returns a copy of the array with its elements randomly redistributed.
	 *
	 * @param array
	 *            the array to randomize.
	 * @param generator
	 *            the random number generator to use
	 * @return a copy of the array with its elements randomly redistributed.
	 */
	private int[] randomize(final int[] array, final Random generator) {

		int index;
		int temp;
		int[] newArray = new int[array.length];
		System.arraycopy(array, 0, newArray, 0, array.length);
		for (int j = newArray.length - 1; j > 0; j--) {
			index = (int) (generator.nextDouble() * j);
			temp = newArray[j];
			newArray[j] = newArray[index];
			newArray[index] = temp;
		}
		return newArray;
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

} // class end
