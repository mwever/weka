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
 *    IBk.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.lazy;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.rules.ZeroR;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 * <!-- globalinfo-start --> K-nearest neighbours classifier. Can select appropriate value of K based on cross-validation. Can also do distance weighting.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * D. Aha, D. Kibler (1991). Instance-based learning algorithms. Machine Learning. 6:37-66.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#64;article{Aha1991,
 *    author = {D. Aha and D. Kibler},
 *    journal = {Machine Learning},
 *    pages = {37-66},
 *    title = {Instance-based learning algorithms},
 *    volume = {6},
 *    year = {1991}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 *  -I
 *  Weight neighbours by the inverse of their distance
 *  (use when k &gt; 1)
 * </pre>
 *
 * <pre>
 *  -F
 *  Weight neighbours by 1 - their distance
 *  (use when k &gt; 1)
 * </pre>
 *
 * <pre>
 *  -K &lt;number of neighbors&gt;
 *  Number of nearest neighbours (k) used in classification.
 *  (Default = 1)
 * </pre>
 *
 * <pre>
 *  -E
 *  Minimise mean squared error rather than mean absolute
 *  error when using -X option with numeric prediction.
 * </pre>
 *
 * <pre>
 *  -W &lt;window size&gt;
 *  Maximum number of training instances maintained.
 *  Training instances are dropped FIFO. (Default = no window)
 * </pre>
 *
 * <pre>
 *  -X
 *  Select the number of nearest neighbours between 1
 *  and the k value specified using hold-one-out evaluation
 *  on the training data (use when k &gt; 1)
 * </pre>
 *
 * <pre>
 *  -A
 *  The nearest neighbour search algorithm to use (default: weka.core.neighboursearch.LinearNNSearch).
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Stuart Inglis (singlis@cs.waikato.ac.nz)
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class IBk extends AbstractClassifier implements OptionHandler, UpdateableClassifier, WeightedInstancesHandler, TechnicalInformationHandler, AdditionalMeasureProducer {

	/** for serialization. */
	static final long serialVersionUID = -3080186098777067172L;

	/** The training instances used for classification. */
	protected Instances m_Train;

	/** The number of class values (or 1 if predicting numeric). */
	protected int m_NumClasses;

	/** The class attribute type. */
	protected int m_ClassType;

	/** The number of neighbours to use for classification (currently). */
	protected int m_kNN;

	/**
	 * The value of kNN provided by the user. This may differ from m_kNN if cross-validation is being used.
	 */
	protected int m_kNNUpper;

	/**
	 * Whether the value of k selected by cross validation has been invalidated by a change in the training instances.
	 */
	protected boolean m_kNNValid;

	/**
	 * The maximum number of training instances allowed. When this limit is reached, old training instances are removed, so the training data is "windowed". Set to 0 for unlimited numbers of instances.
	 */
	protected int m_WindowSize;

	/** Whether the neighbours should be distance-weighted. */
	protected int m_DistanceWeighting;

	/** Whether to select k by cross validation. */
	protected boolean m_CrossValidate;

	/**
	 * Whether to minimise mean squared error rather than mean absolute error when cross-validating on numeric prediction tasks.
	 */
	protected boolean m_MeanSquared;

	/** Default ZeroR model to use when there are no training instances */
	protected ZeroR m_defaultModel;

	/** no weighting. */
	public static final int WEIGHT_NONE = 1;
	/** weight by 1/distance. */
	public static final int WEIGHT_INVERSE = 2;
	/** weight by 1-distance. */
	public static final int WEIGHT_SIMILARITY = 4;
	/** possible instance weighting methods. */
	public static final Tag[] TAGS_WEIGHTING = { new Tag(WEIGHT_NONE, "No distance weighting"), new Tag(WEIGHT_INVERSE, "Weight by 1/distance"), new Tag(WEIGHT_SIMILARITY, "Weight by 1-distance") };

	/** for nearest-neighbor search. */
	protected NearestNeighbourSearch m_NNSearch = new LinearNNSearch();

	/** The number of attributes the contribute to a prediction. */
	protected double m_NumAttributesUsed;

	/**
	 * IBk classifier. Simple instance-based learner that uses the class of the nearest k training instances for the class of the test instances.
	 *
	 * @param k
	 *            the number of nearest neighbors to use for prediction
	 */
	public IBk(final int k) {

		this.init();
		this.setKNN(k);
	}

	/**
	 * IB1 classifer. Instance-based learner. Predicts the class of the single nearest training instance for each test instance.
	 */
	public IBk() {

		this.init();
	}

	/**
	 * Returns a string describing classifier.
	 *
	 * @return a description suitable for displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return "K-nearest neighbours classifier. Can " + "select appropriate value of K based on cross-validation. Can also do " + "distance weighting.\n\n" + "For more information, see\n\n" + this.getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed information about the technical background of this class, e.g., paper reference or book this class is based on.
	 *
	 * @return the technical information about this class
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "D. Aha and D. Kibler");
		result.setValue(Field.YEAR, "1991");
		result.setValue(Field.TITLE, "Instance-based learning algorithms");
		result.setValue(Field.JOURNAL, "Machine Learning");
		result.setValue(Field.VOLUME, "6");
		result.setValue(Field.PAGES, "37-66");

		return result;
	}

	/**
	 * Returns the tip text for this property.
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String KNNTipText() {
		return "The number of neighbours to use.";
	}

	/**
	 * Set the number of neighbours the learner is to use.
	 *
	 * @param k
	 *            the number of neighbours.
	 */
	public void setKNN(final int k) {
		this.m_kNN = k;
		this.m_kNNUpper = k;
		this.m_kNNValid = false;
	}

	/**
	 * Gets the number of neighbours the learner will use.
	 *
	 * @return the number of neighbours.
	 */
	public int getKNN() {

		return this.m_kNN;
	}

	/**
	 * Returns the tip text for this property.
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String windowSizeTipText() {
		return "Gets the maximum number of instances allowed in the training " + "pool. The addition of new instances above this value will result " + "in old instances being removed. A value of 0 signifies no limit "
				+ "to the number of training instances.";
	}

	/**
	 * Gets the maximum number of instances allowed in the training pool. The addition of new instances above this value will result in old instances being removed. A value of 0 signifies no limit to the number of training instances.
	 *
	 * @return Value of WindowSize.
	 */
	public int getWindowSize() {

		return this.m_WindowSize;
	}

	/**
	 * Sets the maximum number of instances allowed in the training pool. The addition of new instances above this value will result in old instances being removed. A value of 0 signifies no limit to the number of training instances.
	 *
	 * @param newWindowSize
	 *            Value to assign to WindowSize.
	 */
	public void setWindowSize(final int newWindowSize) {

		this.m_WindowSize = newWindowSize;
	}

	/**
	 * Returns the tip text for this property.
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String distanceWeightingTipText() {

		return "Gets the distance weighting method used.";
	}

	/**
	 * Gets the distance weighting method used. Will be one of WEIGHT_NONE, WEIGHT_INVERSE, or WEIGHT_SIMILARITY
	 *
	 * @return the distance weighting method used.
	 */
	public SelectedTag getDistanceWeighting() {

		return new SelectedTag(this.m_DistanceWeighting, TAGS_WEIGHTING);
	}

	/**
	 * Sets the distance weighting method used. Values other than WEIGHT_NONE, WEIGHT_INVERSE, or WEIGHT_SIMILARITY will be ignored.
	 *
	 * @param newMethod
	 *            the distance weighting method to use
	 */
	public void setDistanceWeighting(final SelectedTag newMethod) {

		if (newMethod.getTags() == TAGS_WEIGHTING) {
			this.m_DistanceWeighting = newMethod.getSelectedTag().getID();
		}
	}

	/**
	 * Returns the tip text for this property.
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String meanSquaredTipText() {

		return "Whether the mean squared error is used rather than mean " + "absolute error when doing cross-validation for regression problems.";
	}

	/**
	 * Gets whether the mean squared error is used rather than mean absolute error when doing cross-validation.
	 *
	 * @return true if so.
	 */
	public boolean getMeanSquared() {

		return this.m_MeanSquared;
	}

	/**
	 * Sets whether the mean squared error is used rather than mean absolute error when doing cross-validation.
	 *
	 * @param newMeanSquared
	 *            true if so.
	 */
	public void setMeanSquared(final boolean newMeanSquared) {

		this.m_MeanSquared = newMeanSquared;
	}

	/**
	 * Returns the tip text for this property.
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String crossValidateTipText() {

		return "Whether hold-one-out cross-validation will be used to " + "select the best k value between 1 and the value specified as " + "the KNN parameter.";
	}

	/**
	 * Gets whether hold-one-out cross-validation will be used to select the best k value.
	 *
	 * @return true if cross-validation will be used.
	 */
	public boolean getCrossValidate() {

		return this.m_CrossValidate;
	}

	/**
	 * Sets whether hold-one-out cross-validation will be used to select the best k value.
	 *
	 * @param newCrossValidate
	 *            true if cross-validation should be used.
	 */
	public void setCrossValidate(final boolean newCrossValidate) {

		this.m_CrossValidate = newCrossValidate;
	}

	/**
	 * Returns the tip text for this property.
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String nearestNeighbourSearchAlgorithmTipText() {
		return "The nearest neighbour search algorithm to use " + "(Default: weka.core.neighboursearch.LinearNNSearch).";
	}

	/**
	 * Returns the current nearestNeighbourSearch algorithm in use.
	 *
	 * @return the NearestNeighbourSearch algorithm currently in use.
	 */
	public NearestNeighbourSearch getNearestNeighbourSearchAlgorithm() {
		return this.m_NNSearch;
	}

	/**
	 * Sets the nearestNeighbourSearch algorithm to be used for finding nearest neighbour(s).
	 *
	 * @param nearestNeighbourSearchAlgorithm
	 *            - The NearestNeighbourSearch class.
	 */
	public void setNearestNeighbourSearchAlgorithm(final NearestNeighbourSearch nearestNeighbourSearchAlgorithm) {
		this.m_NNSearch = nearestNeighbourSearchAlgorithm;
	}

	/**
	 * Get the number of training instances the classifier is currently using.
	 *
	 * @return the number of training instances the classifier is currently using
	 */
	public int getNumTraining() {

		return this.m_Train.numInstances();
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

		this.m_NumClasses = instances.numClasses();
		this.m_ClassType = instances.classAttribute().type();
		this.m_Train = new Instances(instances, 0, instances.numInstances());

		// Throw away initial instances until within the specified window size
		if ((this.m_WindowSize > 0) && (instances.numInstances() > this.m_WindowSize)) {
			this.m_Train = new Instances(this.m_Train, this.m_Train.numInstances() - this.m_WindowSize, this.m_WindowSize);
		}

		this.m_NumAttributesUsed = 0.0;
		for (int i = 0; i < this.m_Train.numAttributes(); i++) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			if ((i != this.m_Train.classIndex()) && (this.m_Train.attribute(i).isNominal() || this.m_Train.attribute(i).isNumeric())) {
				this.m_NumAttributesUsed += 1.0;
			}
		}

		this.m_NNSearch.setInstances(this.m_Train);

		// Invalidate any currently cross-validation selected k
		this.m_kNNValid = false;

		this.m_defaultModel = new ZeroR();
		this.m_defaultModel.buildClassifier(instances);
	}

	/**
	 * Adds the supplied instance to the training set.
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
		this.m_NNSearch.update(instance);
		this.m_kNNValid = false;
		if ((this.m_WindowSize > 0) && (this.m_Train.numInstances() > this.m_WindowSize)) {
			boolean deletedInstance = false;
			while (this.m_Train.numInstances() > this.m_WindowSize) {
				this.m_Train.delete(0);
				deletedInstance = true;
			}
			// rebuild datastructure KDTree currently can't delete
			if (deletedInstance == true) {
				this.m_NNSearch.setInstances(this.m_Train);
			}
		}
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

		if (this.m_Train.numInstances() == 0) {
			// throw new Exception("No training instances!");
			return this.m_defaultModel.distributionForInstance(instance);
		}
		if ((this.m_WindowSize > 0) && (this.m_Train.numInstances() > this.m_WindowSize)) {
			this.m_kNNValid = false;
			boolean deletedInstance = false;
			while (this.m_Train.numInstances() > this.m_WindowSize) {
				// XXX kill weka execution
				if (Thread.interrupted()) {
					throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
				}
				this.m_Train.delete(0);
			}
			// rebuild datastructure KDTree currently can't delete
			if (deletedInstance == true) {
				this.m_NNSearch.setInstances(this.m_Train);
			}
		}

		// Select k by cross validation
		if (!this.m_kNNValid && (this.m_CrossValidate) && (this.m_kNNUpper >= 1)) {
			this.crossValidate();
		}

		this.m_NNSearch.addInstanceInfo(instance);

		Instances neighbours = this.m_NNSearch.kNearestNeighbours(instance, this.m_kNN);
		double[] distances = this.m_NNSearch.getDistances();
		double[] distribution = this.makeDistribution(neighbours, distances);

		return distribution;
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<>(7);

		newVector.addElement(new Option("\tWeight neighbours by the inverse of their distance\n" + "\t(use when k > 1)", "I", 0, "-I"));
		newVector.addElement(new Option("\tWeight neighbours by 1 - their distance\n" + "\t(use when k > 1)", "F", 0, "-F"));
		newVector.addElement(new Option("\tNumber of nearest neighbours (k) used in classification.\n" + "\t(Default = 1)", "K", 1, "-K <number of neighbors>"));
		newVector.addElement(new Option("\tMinimise mean squared error rather than mean absolute\n" + "\terror when using -X option with numeric prediction.", "E", 0, "-E"));
		newVector.addElement(new Option("\tMaximum number of training instances maintained.\n" + "\tTraining instances are dropped FIFO. (Default = no window)", "W", 1, "-W <window size>"));
		newVector.addElement(new Option("\tSelect the number of nearest neighbours between 1\n" + "\tand the k value specified using hold-one-out evaluation\n" + "\ton the training data (use when k > 1)", "X", 0, "-X"));
		newVector.addElement(new Option("\tThe nearest neighbour search algorithm to use " + "(default: weka.core.neighboursearch.LinearNNSearch).\n", "A", 0, "-A"));

		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 *
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 *
	 * <pre>
	 *  -I
	 *  Weight neighbours by the inverse of their distance
	 *  (use when k &gt; 1)
	 * </pre>
	 *
	 * <pre>
	 *  -F
	 *  Weight neighbours by 1 - their distance
	 *  (use when k &gt; 1)
	 * </pre>
	 *
	 * <pre>
	 *  -K &lt;number of neighbors&gt;
	 *  Number of nearest neighbours (k) used in classification.
	 *  (Default = 1)
	 * </pre>
	 *
	 * <pre>
	 *  -E
	 *  Minimise mean squared error rather than mean absolute
	 *  error when using -X option with numeric prediction.
	 * </pre>
	 *
	 * <pre>
	 *  -W &lt;window size&gt;
	 *  Maximum number of training instances maintained.
	 *  Training instances are dropped FIFO. (Default = no window)
	 * </pre>
	 *
	 * <pre>
	 *  -X
	 *  Select the number of nearest neighbours between 1
	 *  and the k value specified using hold-one-out evaluation
	 *  on the training data (use when k &gt; 1)
	 * </pre>
	 *
	 * <pre>
	 *  -A
	 *  The nearest neighbour search algorithm to use (default: weka.core.neighboursearch.LinearNNSearch).
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

		String knnString = Utils.getOption('K', options);
		if (knnString.length() != 0) {
			this.setKNN(Integer.parseInt(knnString));
		} else {
			this.setKNN(1);
		}
		String windowString = Utils.getOption('W', options);
		if (windowString.length() != 0) {
			this.setWindowSize(Integer.parseInt(windowString));
		} else {
			this.setWindowSize(0);
		}
		if (Utils.getFlag('I', options)) {
			this.setDistanceWeighting(new SelectedTag(WEIGHT_INVERSE, TAGS_WEIGHTING));
		} else if (Utils.getFlag('F', options)) {
			this.setDistanceWeighting(new SelectedTag(WEIGHT_SIMILARITY, TAGS_WEIGHTING));
		} else {
			this.setDistanceWeighting(new SelectedTag(WEIGHT_NONE, TAGS_WEIGHTING));
		}
		this.setCrossValidate(Utils.getFlag('X', options));
		this.setMeanSquared(Utils.getFlag('E', options));

		String nnSearchClass = Utils.getOption('A', options);
		if (nnSearchClass.length() != 0) {
			String nnSearchClassSpec[] = Utils.splitOptions(nnSearchClass);
			if (nnSearchClassSpec.length == 0) {
				throw new Exception("Invalid NearestNeighbourSearch algorithm " + "specification string.");
			}
			String className = nnSearchClassSpec[0];
			nnSearchClassSpec[0] = "";

			this.setNearestNeighbourSearchAlgorithm((NearestNeighbourSearch) Utils.forName(NearestNeighbourSearch.class, className, nnSearchClassSpec));
		} else {
			this.setNearestNeighbourSearchAlgorithm(new LinearNNSearch());
		}

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of IBk.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	@Override
	public String[] getOptions() {

		Vector<String> options = new Vector<>();
		options.add("-K");
		options.add("" + this.getKNN());
		options.add("-W");
		options.add("" + this.m_WindowSize);
		if (this.getCrossValidate()) {
			options.add("-X");
		}
		if (this.getMeanSquared()) {
			options.add("-E");
		}
		if (this.m_DistanceWeighting == WEIGHT_INVERSE) {
			options.add("-I");
		} else if (this.m_DistanceWeighting == WEIGHT_SIMILARITY) {
			options.add("-F");
		}

		options.add("-A");
		options.add(this.m_NNSearch.getClass().getName() + " " + Utils.joinOptions(this.m_NNSearch.getOptions()));

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	}

	/**
	 * Returns an enumeration of the additional measure names produced by the neighbour search algorithm, plus the chosen K in case cross-validation is enabled.
	 *
	 * @return an enumeration of the measure names
	 */
	@Override
	public Enumeration<String> enumerateMeasures() {
		if (this.m_CrossValidate) {
			Enumeration<String> enm = this.m_NNSearch.enumerateMeasures();
			Vector<String> measures = new Vector<>();
			while (enm.hasMoreElements()) {
				measures.add(enm.nextElement());
			}
			measures.add("measureKNN");
			return measures.elements();
		} else {
			return this.m_NNSearch.enumerateMeasures();
		}
	}

	/**
	 * Returns the value of the named measure from the neighbour search algorithm, plus the chosen K in case cross-validation is enabled.
	 *
	 * @param additionalMeasureName
	 *            the name of the measure to query for its value
	 * @return the value of the named measure
	 * @throws IllegalArgumentException
	 *             if the named measure is not supported
	 */
	@Override
	public double getMeasure(final String additionalMeasureName) {
		if (additionalMeasureName.equals("measureKNN")) {
			return this.m_kNN;
		} else {
			return this.m_NNSearch.getMeasure(additionalMeasureName);
		}
	}

	/**
	 * Returns a description of this classifier.
	 *
	 * @return a description of this classifier as a string.
	 */
	@Override
	public String toString() {

		if (this.m_Train == null) {
			return "IBk: No model built yet.";
		}

		if (this.m_Train.numInstances() == 0) {
			return "Warning: no training instances - ZeroR model used.";
		}

		if (!this.m_kNNValid && this.m_CrossValidate) {
			try {
				this.crossValidate();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}

		String result = "IB1 instance-based classifier\n" + "using " + this.m_kNN;

		switch (this.m_DistanceWeighting) {
		case WEIGHT_INVERSE:
			result += " inverse-distance-weighted";
			break;
		case WEIGHT_SIMILARITY:
			result += " similarity-weighted";
			break;
		}
		result += " nearest neighbour(s) for classification\n";

		if (this.m_WindowSize != 0) {
			result += "using a maximum of " + this.m_WindowSize + " (windowed) training instances\n";
		}
		return result;
	}

	/**
	 * Initialise scheme variables.
	 */
	protected void init() {

		this.setKNN(1);
		this.m_WindowSize = 0;
		this.m_DistanceWeighting = WEIGHT_NONE;
		this.m_CrossValidate = false;
		this.m_MeanSquared = false;
	}

	/**
	 * Turn the list of nearest neighbors into a probability distribution.
	 *
	 * @param neighbours
	 *            the list of nearest neighboring instances
	 * @param distances
	 *            the distances of the neighbors
	 * @return the probability distribution
	 * @throws Exception
	 *             if computation goes wrong or has no class attribute
	 */
	protected double[] makeDistribution(final Instances neighbours, final double[] distances) throws Exception {

		double total = 0, weight;
		double[] distribution = new double[this.m_NumClasses];

		// Set up a correction to the estimator
		if (this.m_ClassType == Attribute.NOMINAL) {
			for (int i = 0; i < this.m_NumClasses; i++) {
				distribution[i] = 1.0 / Math.max(1, this.m_Train.numInstances());
			}
			total = (double) this.m_NumClasses / Math.max(1, this.m_Train.numInstances());
		}

		for (int i = 0; i < neighbours.numInstances(); i++) {
			// Collect class counts
			Instance current = neighbours.instance(i);
			distances[i] = distances[i] * distances[i];
			distances[i] = Math.sqrt(distances[i] / this.m_NumAttributesUsed);
			switch (this.m_DistanceWeighting) {
			case WEIGHT_INVERSE:
				weight = 1.0 / (distances[i] + 0.001); // to avoid div by zero
				break;
			case WEIGHT_SIMILARITY:
				weight = 1.0 - distances[i];
				break;
			default: // WEIGHT_NONE:
				weight = 1.0;
				break;
			}
			weight *= current.weight();
			try {
				switch (this.m_ClassType) {
				case Attribute.NOMINAL:
					distribution[(int) current.classValue()] += weight;
					break;
				case Attribute.NUMERIC:
					distribution[0] += current.classValue() * weight;
					break;
				}
			} catch (Exception ex) {
				throw new Error("Data has no class attribute!");
			}
			total += weight;
		}

		// Normalise distribution
		if (total > 0) {
			Utils.normalize(distribution, total);
		}
		return distribution;
	}

	/**
	 * Select the best value for k by hold-one-out cross-validation. If the class attribute is nominal, classification error is minimised. If the class attribute is numeric, mean absolute error is minimised
	 *
	 * @throws InterruptedException
	 */
	protected void crossValidate() throws InterruptedException {

		try {
			if (this.m_NNSearch instanceof weka.core.neighboursearch.CoverTree) {
				throw new Exception("CoverTree doesn't support hold-one-out " + "cross-validation. Use some other NN " + "method.");
			}

			double[] performanceStats = new double[this.m_kNNUpper];
			double[] performanceStatsSq = new double[this.m_kNNUpper];

			for (int i = 0; i < this.m_kNNUpper; i++) {
				performanceStats[i] = 0;
				performanceStatsSq[i] = 0;
			}

			this.m_kNN = this.m_kNNUpper;
			Instance instance;
			Instances neighbours;
			double[] origDistances, convertedDistances;
			for (int i = 0; i < this.m_Train.numInstances(); i++) {
				// XXX kill weka execution
				if (Thread.interrupted()) {
					throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
				}
				if (this.m_Debug && (i % 50 == 0)) {
					System.err.print("Cross validating " + i + "/" + this.m_Train.numInstances() + "\r");
				}
				instance = this.m_Train.instance(i);
				neighbours = this.m_NNSearch.kNearestNeighbours(instance, this.m_kNN);
				origDistances = this.m_NNSearch.getDistances();

				for (int j = this.m_kNNUpper - 1; j >= 0; j--) {
					// XXX kill weka execution
					if (Thread.interrupted()) {
						throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
					}
					// Update the performance stats
					convertedDistances = new double[origDistances.length];
					System.arraycopy(origDistances, 0, convertedDistances, 0, origDistances.length);
					double[] distribution = this.makeDistribution(neighbours, convertedDistances);
					double thisPrediction = Utils.maxIndex(distribution);
					if (this.m_Train.classAttribute().isNumeric()) {
						thisPrediction = distribution[0];
						double err = thisPrediction - instance.classValue();
						performanceStatsSq[j] += err * err; // Squared error
						performanceStats[j] += Math.abs(err); // Absolute error
					} else {
						if (thisPrediction != instance.classValue()) {
							performanceStats[j]++; // Classification error
						}
					}
					if (j >= 1) {
						neighbours = this.pruneToK(neighbours, convertedDistances, j);
					}
				}
			}

			// Display the results of the cross-validation
			for (int i = 0; i < this.m_kNNUpper; i++) {
				if (this.m_Debug) {
					System.err.print("Hold-one-out performance of " + (i + 1) + " neighbors ");
				}
				if (this.m_Train.classAttribute().isNumeric()) {
					if (this.m_Debug) {
						if (this.m_MeanSquared) {
							System.err.println("(RMSE) = " + Math.sqrt(performanceStatsSq[i] / this.m_Train.numInstances()));
						} else {
							System.err.println("(MAE) = " + performanceStats[i] / this.m_Train.numInstances());
						}
					}
				} else {
					if (this.m_Debug) {
						System.err.println("(%ERR) = " + 100.0 * performanceStats[i] / this.m_Train.numInstances());
					}
				}
			}

			// Check through the performance stats and select the best
			// k value (or the lowest k if more than one best)
			double[] searchStats = performanceStats;
			if (this.m_Train.classAttribute().isNumeric() && this.m_MeanSquared) {
				searchStats = performanceStatsSq;
			}
			double bestPerformance = Double.NaN;
			int bestK = 1;
			for (int i = 0; i < this.m_kNNUpper; i++) {
				if (Double.isNaN(bestPerformance) || (bestPerformance > searchStats[i])) {
					bestPerformance = searchStats[i];
					bestK = i + 1;
				}
			}
			this.m_kNN = bestK;
			if (this.m_Debug) {
				System.err.println("Selected k = " + bestK);
			}

			this.m_kNNValid = true;
		} catch (InterruptedException e) {
			throw e;
		} catch (Exception ex) {
			throw new Error("Couldn't optimize by cross-validation: " + ex.getMessage());
		}
	}

	/**
	 * Prunes the list to contain the k nearest neighbors. If there are multiple neighbors at the k'th distance, all will be kept.
	 *
	 * @param neighbours
	 *            the neighbour instances.
	 * @param distances
	 *            the distances of the neighbours from target instance.
	 * @param k
	 *            the number of neighbors to keep.
	 * @return the pruned neighbours.
	 */
	public Instances pruneToK(Instances neighbours, final double[] distances, int k) {

		if (neighbours == null || distances == null || neighbours.numInstances() == 0) {
			return null;
		}
		if (k < 1) {
			k = 1;
		}

		int currentK = 0;
		double currentDist;
		for (int i = 0; i < neighbours.numInstances(); i++) {
			currentK++;
			currentDist = distances[i];
			if (currentK > k && currentDist != distances[i - 1]) {
				currentK--;
				neighbours = new Instances(neighbours, 0, currentK);
				break;
			}
		}

		return neighbours;
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

	/**
	 * Main method for testing this class.
	 *
	 * @param argv
	 *            should contain command line options (see setOptions)
	 */
	public static void main(final String[] argv) {
		runClassifier(new IBk(), argv);
	}
}
