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
 *    LinearRegression.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.functions;

import java.util.Collections;
import java.util.Enumeration;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.UpperSymmDenseMatrix;
import no.uib.cipr.matrix.Vector;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.RegressionAnalysis;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * <!-- globalinfo-start --> Class for using linear regression for prediction. Uses the Akaike
 * criterion for model selection, and is able to deal with weighted instances.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -S &lt;number of selection method&gt;
 *  Set the attribute selection method to use. 1 = None, 2 = Greedy.
 *  (default 0 = M5' method)
 * </pre>
 *
 * <pre>
 * -C
 *  Do not try to eliminate colinear attributes.
 * </pre>
 *
 * <pre>
 * -R &lt;double&gt;
 *  Set ridge parameter (default 1.0e-8).
 * </pre>
 *
 * <pre>
 * -minimal
 *  Conserve memory, don't keep dataset header and means/stdevs.
 *  Model cannot be printed out if this option is enabled. (default: keep data)
 * </pre>
 *
 * <pre>
 * -additional-stats
 *  Output additional statistics.
 * </pre>
 *
 * <pre>
 * -output-debug-info
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 *
 * <pre>
 * -do-not-check-capabilities
 *  If set, classifier capabilities are not checked before classifier is built
 *  (use with caution).
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class LinearRegression extends AbstractClassifier implements OptionHandler, WeightedInstancesHandler {

	/** Attribute selection method: M5 method */
	public static final int SELECTION_M5 = 0;
	/** Attribute selection method: No attribute selection */
	public static final int SELECTION_NONE = 1;
	/** Attribute selection method: Greedy method */
	public static final int SELECTION_GREEDY = 2;
	/** Attribute selection methods */
	public static final Tag[] TAGS_SELECTION = { new Tag(SELECTION_NONE, "No attribute selection"), new Tag(SELECTION_M5, "M5 method"), new Tag(SELECTION_GREEDY, "Greedy method") };
	/** for serialization */
	static final long serialVersionUID = -3364580862046573747L;
	/** Array for storing coefficients of linear regression. */
	protected double[] m_Coefficients;
	/** Which attributes are relevant? */
	protected boolean[] m_SelectedAttributes;
	/** Variable for storing transformed training data. */
	protected Instances m_TransformedData;
	/** The filter for removing missing values. */
	protected ReplaceMissingValues m_MissingFilter;
	/**
	 * The filter storing the transformation from nominal to binary attributes.
	 */
	protected NominalToBinary m_TransformFilter;
	/** The standard deviations of the class attribute */
	protected double m_ClassStdDev;
	/** The mean of the class attribute */
	protected double m_ClassMean;
	/** The index of the class attribute */
	protected int m_ClassIndex;
	/** The attributes means */
	protected double[] m_Means;
	/** The attribute standard deviations */
	protected double[] m_StdDevs;
	/**
	 * Whether to output additional statistics such as std. dev. of coefficients and t-stats
	 */
	protected boolean m_outputAdditionalStats;
	/** The current attribute selection method */
	protected int m_AttributeSelection;
	/** Try to eliminate correlated attributes? */
	protected boolean m_EliminateColinearAttributes = true;
	/** Turn off all checks and conversions? */
	protected boolean m_checksTurnedOff = false;
	/** The ridge parameter */
	protected double m_Ridge = 1.0e-8;
	/** Conserve memory? */
	protected boolean m_Minimal = false;
	/** Model already built? */
	protected boolean m_ModelBuilt = false;
	/** True if the model is a zero R one */
	protected boolean m_isZeroR;
	/** The degrees of freedom of the regression model */
	private int m_df;
	/** The R-squared value of the regression model */
	private double m_RSquared;
	/** The adjusted R-squared value of the regression model */
	private double m_RSquaredAdj;
	/** The F-statistic of the regression model */
	private double m_FStat;
	/** Array for storing the standard error of each coefficient */
	private double[] m_StdErrorOfCoef;
	/** Array for storing the t-statistic of each coefficient */
	private double[] m_TStats;

	public LinearRegression() {
		this.m_numDecimalPlaces = 4;
	}

	/**
	 * Generates a linear regression function predictor.
	 *
	 * @param argv
	 *          the options
	 */
	public static void main(final String argv[]) {
		runClassifier(new LinearRegression(), argv);
	}

	/**
	 * Returns a string describing this classifier
	 *
	 * @return a description of the classifier suitable for displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Class for using linear regression for prediction. Uses the Akaike " + "criterion for model selection, and is able to deal with weighted " + "instances.";
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
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.DATE_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		return result;
	}

	/**
	 * Builds a regression model for the given data.
	 *
	 * @param data
	 *          the training data to be used for generating the linear regression function
	 * @throws Exception
	 *           if the classifier could not be built successfully
	 */
	@Override
	public void buildClassifier(Instances data) throws Exception {
		this.m_ModelBuilt = false;
		this.m_isZeroR = false;

		if (data.numInstances() == 1) {
			this.m_Coefficients = new double[1];
			this.m_Coefficients[0] = data.instance(0).classValue();
			this.m_SelectedAttributes = new boolean[data.numAttributes()];
			this.m_isZeroR = true;
			return;
		}

		if (!this.m_checksTurnedOff) {
			// can classifier handle the data?
			this.getCapabilities().testWithFail(data);

			if (this.m_outputAdditionalStats) {
				// check that the instances weights are all 1
				// because the RegressionAnalysis class does
				// not handle weights
				boolean ok = true;
				for (int i = 0; i < data.numInstances(); i++) {
					// XXX kill weka execution
					if (Thread.interrupted()) {
						throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
					}
					if (data.instance(i).weight() != 1) {
						ok = false;
						break;
					}
				}
				if (!ok) {
					throw new Exception("Can only compute additional statistics on unweighted data");
				}
			}

			// remove instances with missing class
			data = new Instances(data);
			data.deleteWithMissingClass();

			this.m_TransformFilter = new NominalToBinary();
			this.m_TransformFilter.setInputFormat(data);
			data = Filter.useFilter(data, this.m_TransformFilter);
			this.m_MissingFilter = new ReplaceMissingValues();
			this.m_MissingFilter.setInputFormat(data);
			data = Filter.useFilter(data, this.m_MissingFilter);
			data.deleteWithMissingClass();
		} else {
			this.m_TransformFilter = null;
			this.m_MissingFilter = null;
		}

		this.m_ClassIndex = data.classIndex();
		this.m_TransformedData = data;

		// Turn all attributes on for a start
		this.m_Coefficients = null;

		// Compute means and standard deviations
		this.m_SelectedAttributes = new boolean[data.numAttributes()];
		this.m_Means = new double[data.numAttributes()];
		this.m_StdDevs = new double[data.numAttributes()];
		for (int j = 0; j < data.numAttributes(); j++) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			if (j != this.m_ClassIndex) {
				this.m_SelectedAttributes[j] = true; // Turn attributes on for a start
				this.m_Means[j] = data.meanOrMode(j);
				this.m_StdDevs[j] = Math.sqrt(data.variance(j));
				if (this.m_StdDevs[j] == 0) {
					this.m_SelectedAttributes[j] = false;
				}
			}
		}

		this.m_ClassStdDev = Math.sqrt(data.variance(this.m_TransformedData.classIndex()));
		this.m_ClassMean = data.meanOrMode(this.m_TransformedData.classIndex());

		// Perform the regression
		this.findBestModel();

		if (this.m_outputAdditionalStats) {
			// find number of coefficients, degrees of freedom
			int k = 1;
			for (int i = 0; i < data.numAttributes(); i++) {
				if (i != data.classIndex()) {
					if (this.m_SelectedAttributes[i]) {
						k++;
					}
				}
			}
			this.m_df = this.m_TransformedData.numInstances() - k;

			// calculate R^2 and F-stat
			double se = this.calculateSE(this.m_SelectedAttributes, this.m_Coefficients);
			this.m_RSquared = RegressionAnalysis.calculateRSquared(this.m_TransformedData, se);
			this.m_RSquaredAdj = RegressionAnalysis.calculateAdjRSquared(this.m_RSquared, this.m_TransformedData.numInstances(), k);
			this.m_FStat = RegressionAnalysis.calculateFStat(this.m_RSquared, this.m_TransformedData.numInstances(), k);
			// calculate std error of coefficients and t-stats
			this.m_StdErrorOfCoef = RegressionAnalysis.calculateStdErrorOfCoef(this.m_TransformedData, this.m_SelectedAttributes, se, this.m_TransformedData.numInstances(), k);
			this.m_TStats = RegressionAnalysis.calculateTStats(this.m_Coefficients, this.m_StdErrorOfCoef, k);
		}

		// Save memory
		if (this.m_Minimal) {
			this.m_TransformedData = null;
			this.m_Means = null;
			this.m_StdDevs = null;
		} else {
			this.m_TransformedData = new Instances(data, 0);
		}

		this.m_ModelBuilt = true;
	}

	/**
	 * Classifies the given instance using the linear regression function.
	 *
	 * @param instance
	 *          the test instance
	 * @return the classification
	 * @throws Exception
	 *           if classification can't be done successfully
	 */
	@Override
	public double classifyInstance(final Instance instance) throws Exception {

		// Transform the input instance
		Instance transformedInstance = instance;
		if (!this.m_checksTurnedOff && !this.m_isZeroR) {
			this.m_TransformFilter.input(transformedInstance);
			this.m_TransformFilter.batchFinished();
			transformedInstance = this.m_TransformFilter.output();
			this.m_MissingFilter.input(transformedInstance);
			this.m_MissingFilter.batchFinished();
			transformedInstance = this.m_MissingFilter.output();
		}

		// Calculate the dependent variable from the regression model
		return this.regressionPrediction(transformedInstance, this.m_SelectedAttributes, this.m_Coefficients);
	}

	/**
	 * Outputs the linear regression model as a string.
	 *
	 * @return the model as string
	 */
	@Override
	public String toString() {
		if (!this.m_ModelBuilt) {
			return "Linear Regression: No model built yet.";
		}

		if (this.m_Minimal) {
			return "Linear Regression: Model built.";
		}

		try {
			StringBuilder text = new StringBuilder();
			int column = 0;
			boolean first = true;

			text.append("\nLinear Regression Model\n\n");

			text.append(this.m_TransformedData.classAttribute().name() + " =\n\n");
			for (int i = 0; i < this.m_TransformedData.numAttributes(); i++) {
				if ((i != this.m_ClassIndex) && (this.m_SelectedAttributes[i])) {
					if (!first) {
						text.append(" +\n");
					} else {
						first = false;
					}
					text.append(Utils.doubleToString(this.m_Coefficients[column], 12, this.m_numDecimalPlaces) + " * ");
					text.append(this.m_TransformedData.attribute(i).name());
					column++;
				}
			}
			text.append(" +\n" + Utils.doubleToString(this.m_Coefficients[column], 12, this.m_numDecimalPlaces));

			if (this.m_outputAdditionalStats) {
				int maxAttLength = 0;
				for (int i = 0; i < this.m_TransformedData.numAttributes(); i++) {
					if ((i != this.m_ClassIndex) && (this.m_SelectedAttributes[i])) {
						if (this.m_TransformedData.attribute(i).name().length() > maxAttLength) {
							maxAttLength = this.m_TransformedData.attribute(i).name().length();
						}
					}
				}
				maxAttLength += 3;
				if (maxAttLength < "Variable".length() + 3) {
					maxAttLength = "Variable".length() + 3;
				}

				text.append("\n\nRegression Analysis:\n\n" + Utils.padRight("Variable", maxAttLength) + "  Coefficient     SE of Coef        t-Stat");
				column = 0;
				for (int i = 0; i < this.m_TransformedData.numAttributes(); i++) {
					if ((i != this.m_ClassIndex) && (this.m_SelectedAttributes[i])) {
						text.append("\n" + Utils.padRight(this.m_TransformedData.attribute(i).name(), maxAttLength));
						text.append(Utils.doubleToString(this.m_Coefficients[column], 12, this.m_numDecimalPlaces));
						text.append("   " + Utils.doubleToString(this.m_StdErrorOfCoef[column], 12, this.m_numDecimalPlaces));
						text.append("   " + Utils.doubleToString(this.m_TStats[column], 12, this.m_numDecimalPlaces));
						column++;
					}
				}
				text.append(Utils.padRight("\nconst", maxAttLength + 1) + Utils.doubleToString(this.m_Coefficients[column], 12, this.m_numDecimalPlaces));
				text.append("   " + Utils.doubleToString(this.m_StdErrorOfCoef[column], 12, this.m_numDecimalPlaces));
				text.append("   " + Utils.doubleToString(this.m_TStats[column], 12, this.m_numDecimalPlaces));

				text.append("\n\nDegrees of freedom = " + Integer.toString(this.m_df));
				text.append("\nR^2 value = " + Utils.doubleToString(this.m_RSquared, this.m_numDecimalPlaces));
				text.append("\nAdjusted R^2 = " + Utils.doubleToString(this.m_RSquaredAdj, 5));
				text.append("\nF-statistic = " + Utils.doubleToString(this.m_FStat, this.m_numDecimalPlaces));
			}

			return text.toString();
		} catch (Exception e) {
			return "Can't print Linear Regression!";
		}
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		java.util.Vector<Option> newVector = new java.util.Vector<>();

		newVector.addElement(new Option("\tSet the attribute selection method" + " to use. 1 = None, 2 = Greedy.\n" + "\t(default 0 = M5' method)", "S", 1, "-S <number of selection method>"));

		newVector.addElement(new Option("\tDo not try to eliminate colinear" + " attributes.\n", "C", 0, "-C"));

		newVector.addElement(new Option("\tSet ridge parameter (default 1.0e-8).\n", "R", 1, "-R <double>"));

		newVector.addElement(new Option("\tConserve memory, don't keep dataset header and means/stdevs.\n" + "\tModel cannot be printed out if this option is enabled." + "\t(default: keep data)", "minimal", 0, "-minimal"));

		newVector.addElement(new Option("\tOutput additional statistics.", "additional-stats", 0, "-additional-stats"));

		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	}

	/**
	 * Returns the coefficients for this linear model.
	 *
	 * @return the coefficients for this linear model
	 */
	public double[] coefficients() {

		double[] coefficients = new double[this.m_SelectedAttributes.length + 1];
		int counter = 0;
		for (int i = 0; i < this.m_SelectedAttributes.length; i++) {
			if ((this.m_SelectedAttributes[i]) && ((i != this.m_ClassIndex))) {
				coefficients[i] = this.m_Coefficients[counter++];
			}
		}
		coefficients[this.m_SelectedAttributes.length] = this.m_Coefficients[counter];
		return coefficients;
	}

	/**
	 * Gets the current settings of the classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {
		java.util.Vector<String> result = new java.util.Vector<>();

		result.add("-S");
		result.add("" + this.getAttributeSelectionMethod().getSelectedTag().getID());

		if (!this.getEliminateColinearAttributes()) {
			result.add("-C");
		}

		result.add("-R");
		result.add("" + this.getRidge());

		if (this.getMinimal()) {
			result.add("-minimal");
		}

		if (this.getOutputAdditionalStats()) {
			result.add("-additional-stats");
		}

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 *
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 *
	 * <pre>
	 * -S &lt;number of selection method&gt;
	 *  Set the attribute selection method to use. 1 = None, 2 = Greedy.
	 *  (default 0 = M5' method)
	 * </pre>
	 *
	 * <pre>
	 * -C
	 *  Do not try to eliminate colinear attributes.
	 * </pre>
	 *
	 * <pre>
	 * -R &lt;double&gt;
	 *  Set ridge parameter (default 1.0e-8).
	 * </pre>
	 *
	 * <pre>
	 * -minimal
	 *  Conserve memory, don't keep dataset header and means/stdevs.
	 *  Model cannot be printed out if this option is enabled. (default: keep data)
	 * </pre>
	 *
	 * <pre>
	 * -additional-stats
	 *  Output additional statistics.
	 * </pre>
	 *
	 * <pre>
	 * -output-debug-info
	 *  If set, classifier is run in debug mode and
	 *  may output additional info to the console
	 * </pre>
	 *
	 * <pre>
	 * -do-not-check-capabilities
	 *  If set, classifier capabilities are not checked before classifier is built
	 *  (use with caution).
	 * </pre>
	 *
	 * <!-- options-end -->
	 *
	 * @param options
	 *          the list of options as an array of strings
	 * @throws Exception
	 *           if an option is not supported
	 */
	@Override
	public void setOptions(final String[] options) throws Exception {

		String selectionString = Utils.getOption('S', options);
		if (selectionString.length() != 0) {
			this.setAttributeSelectionMethod(new SelectedTag(Integer.parseInt(selectionString), TAGS_SELECTION));
		} else {
			this.setAttributeSelectionMethod(new SelectedTag(SELECTION_M5, TAGS_SELECTION));
		}
		String ridgeString = Utils.getOption('R', options);
		if (ridgeString.length() != 0) {
			this.setRidge(new Double(ridgeString).doubleValue());
		} else {
			this.setRidge(1.0e-8);
		}
		this.setEliminateColinearAttributes(!Utils.getFlag('C', options));
		this.setMinimal(Utils.getFlag("minimal", options));

		this.setOutputAdditionalStats(Utils.getFlag("additional-stats", options));

		super.setOptions(options);
		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String ridgeTipText() {
		return "The value of the Ridge parameter.";
	}

	/**
	 * Get the value of Ridge.
	 *
	 * @return Value of Ridge.
	 */
	public double getRidge() {

		return this.m_Ridge;
	}

	/**
	 * Set the value of Ridge.
	 *
	 * @param newRidge
	 *          Value to assign to Ridge.
	 */
	public void setRidge(final double newRidge) {

		this.m_Ridge = newRidge;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String eliminateColinearAttributesTipText() {
		return "Eliminate colinear attributes.";
	}

	/**
	 * Get the value of EliminateColinearAttributes.
	 *
	 * @return Value of EliminateColinearAttributes.
	 */
	public boolean getEliminateColinearAttributes() {

		return this.m_EliminateColinearAttributes;
	}

	/**
	 * Set the value of EliminateColinearAttributes.
	 *
	 * @param newEliminateColinearAttributes
	 *          Value to assign to EliminateColinearAttributes.
	 */
	public void setEliminateColinearAttributes(final boolean newEliminateColinearAttributes) {

		this.m_EliminateColinearAttributes = newEliminateColinearAttributes;
	}

	/**
	 * Get the number of coefficients used in the model
	 *
	 * @return the number of coefficients
	 */
	public int numParameters() {
		return this.m_Coefficients.length - 1;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String attributeSelectionMethodTipText() {
		return "Set the method used to select attributes for use in the linear " + "regression. Available methods are: no attribute selection, attribute " + "selection using M5's method (step through the attributes removing the one "
				+ "with the smallest standardised coefficient until no improvement is observed " + "in the estimate of the error given by the Akaike " + "information criterion), and a greedy selection using the Akaike information "
				+ "metric.";
	}

	/**
	 * Gets the method used to select attributes for use in the linear regression.
	 *
	 * @return the method to use.
	 */
	public SelectedTag getAttributeSelectionMethod() {

		return new SelectedTag(this.m_AttributeSelection, TAGS_SELECTION);
	}

	/**
	 * Sets the method used to select attributes for use in the linear regression.
	 *
	 * @param method
	 *          the attribute selection method to use.
	 */
	public void setAttributeSelectionMethod(final SelectedTag method) {

		if (method.getTags() == TAGS_SELECTION) {
			this.m_AttributeSelection = method.getSelectedTag().getID();
		}
	}

	/**
	 * Returns the tip text for this property.
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String minimalTipText() {
		return "If enabled, dataset header, means and stdevs get discarded to conserve memory; also, the model cannot be printed out.";
	}

	/**
	 * Returns whether to be more memory conservative or being able to output the model as string.
	 *
	 * @return true if memory conservation is preferred over outputting model description
	 */
	public boolean getMinimal() {
		return this.m_Minimal;
	}

	/**
	 * Sets whether to be more memory conservative or being able to output the model as string.
	 *
	 * @param value
	 *          if true memory will be conserved
	 */
	public void setMinimal(final boolean value) {
		this.m_Minimal = value;
	}

	/**
	 * Returns the tip text for this property.
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String outputAdditionalStatsTipText() {
		return "Output additional statistics (such as " + "std deviation of coefficients and t-statistics)";
	}

	/**
	 * Get whether to output additional statistics (such as std. deviation of coefficients and
	 * t-statistics
	 *
	 * @return true if additional stats are to be output
	 */
	public boolean getOutputAdditionalStats() {
		return this.m_outputAdditionalStats;
	}

	/**
	 * Set whether to output additional statistics (such as std. deviation of coefficients and
	 * t-statistics
	 *
	 * @param additional
	 *          true if additional stats are to be output
	 */
	public void setOutputAdditionalStats(final boolean additional) {
		this.m_outputAdditionalStats = additional;
	}

	/**
	 * Turns off checks for missing values, etc. Use with caution. Also turns off scaling.
	 */
	public void turnChecksOff() {
		this.m_checksTurnedOff = true;
	}

	/**
	 * Turns on checks for missing values, etc. Also turns on scaling.
	 */
	public void turnChecksOn() {
		this.m_checksTurnedOff = false;
	}

	/**
	 * Removes the attribute with the highest standardised coefficient greater than 1.5 from the
	 * selected attributes.
	 *
	 * @param selectedAttributes
	 *          an array of flags indicating which attributes are included in the regression model
	 * @param coefficients
	 *          an array of coefficients for the regression model
	 * @return true if an attribute was removed
	 */
	protected boolean deselectColinearAttributes(final boolean[] selectedAttributes, final double[] coefficients) {

		double maxSC = 1.5;
		int maxAttr = -1, coeff = 0;
		for (int i = 0; i < selectedAttributes.length; i++) {
			if (selectedAttributes[i]) {
				double SC = Math.abs(coefficients[coeff] * this.m_StdDevs[i] / this.m_ClassStdDev);
				if (SC > maxSC) {
					maxSC = SC;
					maxAttr = i;
				}
				coeff++;
			}
		}
		if (maxAttr >= 0) {
			selectedAttributes[maxAttr] = false;
			if (this.m_Debug) {
				System.out.println("Deselected colinear attribute:" + (maxAttr + 1) + " with standardised coefficient: " + maxSC);
			}
			return true;
		}
		return false;
	}

	/**
	 * Performs a greedy search for the best regression model using Akaike's criterion.
	 *
	 * @throws Exception
	 *           if regression can't be done
	 */
	protected void findBestModel() throws Exception {

		// For the weighted case we still use numInstances in
		// the calculation of the Akaike criterion.
		int numInstances = this.m_TransformedData.numInstances();

		if (this.m_Debug) {
			System.out.println((new Instances(this.m_TransformedData, 0)).toString());
		}

		// Perform a regression for the full model, and remove colinear attributes
		do {
			this.m_Coefficients = this.doRegression(this.m_SelectedAttributes);
		} while (this.m_EliminateColinearAttributes && this.deselectColinearAttributes(this.m_SelectedAttributes, this.m_Coefficients));

		// Figure out current number of attributes + 1. (We treat this model
		// as the full model for the Akaike-based methods.)
		int numAttributes = 1;
		for (boolean m_SelectedAttribute : this.m_SelectedAttributes) {
			if (m_SelectedAttribute) {
				numAttributes++;
			}
		}

		double fullMSE = this.calculateSE(this.m_SelectedAttributes, this.m_Coefficients);
		double akaike = (numInstances - numAttributes) + 2 * numAttributes;
		if (this.m_Debug) {
			System.out.println("Initial Akaike value: " + akaike);
		}

		boolean improved;
		int currentNumAttributes = numAttributes;
		switch (this.m_AttributeSelection) {

		case SELECTION_GREEDY:

			// Greedy attribute removal
			do {
				boolean[] currentSelected = this.m_SelectedAttributes.clone();
				improved = false;
				currentNumAttributes--;

				for (int i = 0; i < this.m_SelectedAttributes.length; i++) {
					if (currentSelected[i]) {

						// Calculate the akaike rating without this attribute
						currentSelected[i] = false;
						double[] currentCoeffs = this.doRegression(currentSelected);
						double currentMSE = this.calculateSE(currentSelected, currentCoeffs);
						double currentAkaike = currentMSE / fullMSE * (numInstances - numAttributes) + 2 * currentNumAttributes;
						if (this.m_Debug) {
							System.out.println("(akaike: " + currentAkaike);
						}

						// If it is better than the current best
						if (currentAkaike < akaike) {
							if (this.m_Debug) {
								System.err.println("Removing attribute " + (i + 1) + " improved Akaike: " + currentAkaike);
							}
							improved = true;
							akaike = currentAkaike;
							System.arraycopy(currentSelected, 0, this.m_SelectedAttributes, 0, this.m_SelectedAttributes.length);
							this.m_Coefficients = currentCoeffs;
						}
						currentSelected[i] = true;
					}
				}
			} while (improved);
			break;

		case SELECTION_M5:

			// Step through the attributes removing the one with the smallest
			// standardised coefficient until no improvement in Akaike
			do {
				improved = false;
				currentNumAttributes--;

				// Find attribute with smallest SC
				double minSC = 0;
				int minAttr = -1, coeff = 0;
				for (int i = 0; i < this.m_SelectedAttributes.length; i++) {
					if (this.m_SelectedAttributes[i]) {
						double SC = Math.abs(this.m_Coefficients[coeff] * this.m_StdDevs[i] / this.m_ClassStdDev);
						if ((coeff == 0) || (SC < minSC)) {
							minSC = SC;
							minAttr = i;
						}
						coeff++;
					}
				}

				// See whether removing it improves the Akaike score
				if (minAttr >= 0) {
					this.m_SelectedAttributes[minAttr] = false;
					double[] currentCoeffs = this.doRegression(this.m_SelectedAttributes);
					double currentMSE = this.calculateSE(this.m_SelectedAttributes, currentCoeffs);
					double currentAkaike = currentMSE / fullMSE * (numInstances - numAttributes) + 2 * currentNumAttributes;
					if (this.m_Debug) {
						System.out.println("(akaike: " + currentAkaike);
					}

					// If it is better than the current best
					if (currentAkaike < akaike) {
						if (this.m_Debug) {
							System.err.println("Removing attribute " + (minAttr + 1) + " improved Akaike: " + currentAkaike);
						}
						improved = true;
						akaike = currentAkaike;
						this.m_Coefficients = currentCoeffs;
					} else {
						this.m_SelectedAttributes[minAttr] = true;
					}
				}
			} while (improved);
			break;

		case SELECTION_NONE:
			break;
		}
	}

	/**
	 * Calculate the squared error of a regression model on the training data
	 *
	 * @param selectedAttributes
	 *          an array of flags indicating which attributes are included in the regression model
	 * @param coefficients
	 *          an array of coefficients for the regression model
	 * @return the mean squared error on the training data
	 * @throws Exception
	 *           if there is a missing class value in the training data
	 */
	protected double calculateSE(final boolean[] selectedAttributes, final double[] coefficients) throws Exception {

		double mse = 0;
		for (int i = 0; i < this.m_TransformedData.numInstances(); i++) {
			double prediction = this.regressionPrediction(this.m_TransformedData.instance(i), selectedAttributes, coefficients);
			double error = prediction - this.m_TransformedData.instance(i).classValue();
			mse += error * error;
		}
		return mse;
	}

	/**
	 * Calculate the dependent value for a given instance for a given regression model.
	 *
	 * @param transformedInstance
	 *          the input instance
	 * @param selectedAttributes
	 *          an array of flags indicating which attributes are included in the regression model
	 * @param coefficients
	 *          an array of coefficients for the regression model
	 * @return the regression value for the instance.
	 * @throws Exception
	 *           if the class attribute of the input instance is not assigned
	 */
	protected double regressionPrediction(final Instance transformedInstance, final boolean[] selectedAttributes, final double[] coefficients) throws Exception {

		double result = 0;
		int column = 0;
		for (int j = 0; j < transformedInstance.numAttributes(); j++) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			if ((this.m_ClassIndex != j) && (selectedAttributes[j])) {
				result += coefficients[column] * transformedInstance.value(j);
				column++;
			}
		}
		result += coefficients[column];

		return result;
	}

	/**
	 * Calculate a linear regression using the selected attributes
	 *
	 * @param selectedAttributes
	 *          an array of booleans where each element is true if the corresponding attribute should be
	 *          included in the regression.
	 * @return an array of coefficients for the linear regression model.
	 * @throws Exception
	 *           if an error occurred during the regression.
	 */
	protected double[] doRegression(final boolean[] selectedAttributes) throws Exception {

		if (this.m_Debug) {
			System.out.print("doRegression(");
			for (boolean selectedAttribute : selectedAttributes) {
				System.out.print(" " + selectedAttribute);
			}
			System.out.println(" )");
		}
		int numAttributes = 0;
		for (boolean selectedAttribute : selectedAttributes) {
			if (selectedAttribute) {
				numAttributes++;
			}
		}

		// Check whether there are still attributes left
		Matrix independentTransposed = null;
		Vector dependent = null;
		if (numAttributes > 0) {
			independentTransposed = new DenseMatrix(numAttributes, this.m_TransformedData.numInstances());
			dependent = new DenseVector(this.m_TransformedData.numInstances());
			for (int i = 0; i < this.m_TransformedData.numInstances(); i++) {
				Instance inst = this.m_TransformedData.instance(i);
				double sqrt_weight = Math.sqrt(inst.weight());
				int row = 0;
				for (int j = 0; j < this.m_TransformedData.numAttributes(); j++) {
					if (j == this.m_ClassIndex) {
						dependent.set(i, inst.classValue() * sqrt_weight);
					} else {
						if (selectedAttributes[j]) {
							double value = inst.value(j) - this.m_Means[j];

							// We only need to do this if we want to
							// scale the input
							if (!this.m_checksTurnedOff) {
								value /= this.m_StdDevs[j];
							}
							independentTransposed.set(row, i, value * sqrt_weight);
							row++;
						}
					}
				}
			}
		}

		// Compute coefficients (note that we have to treat the
		// intercept separately so that it doesn't get affected
		// by the ridge constant.)
		double[] coefficients = new double[numAttributes + 1];
		if (numAttributes > 0) {

			Vector aTy = independentTransposed.mult(dependent, new DenseVector(numAttributes));
			Matrix aTa = new UpperSymmDenseMatrix(numAttributes).rank1(independentTransposed);
			independentTransposed = null;
			dependent = null;

			boolean success = true;
			Vector coeffsWithoutIntercept = null;
			double ridge = this.getRidge();
			do {
				for (int i = 0; i < numAttributes; i++) {
					aTa.add(i, i, ridge);
				}
				try {
					coeffsWithoutIntercept = aTa.solve(aTy, new DenseVector(numAttributes));
					success = true;
				} catch (Exception ex) {
					for (int i = 0; i < numAttributes; i++) {
						aTa.add(i, i, -ridge);
					}
					ridge *= 10;
					success = false;
				}
			} while (!success);

			System.arraycopy(((DenseVector) coeffsWithoutIntercept).getData(), 0, coefficients, 0, numAttributes);
		}
		coefficients[numAttributes] = this.m_ClassMean;

		// Convert coefficients into original scale
		int column = 0;
		for (int i = 0; i < this.m_TransformedData.numAttributes(); i++) {
			if ((i != this.m_TransformedData.classIndex()) && (selectedAttributes[i])) {

				// We only need to do this if we have scaled the
				// input.
				if (!this.m_checksTurnedOff) {
					coefficients[column] /= this.m_StdDevs[i];
				}

				// We have centred the input
				coefficients[coefficients.length - 1] -= coefficients[column] * this.m_Means[i];
				column++;
			}
		}

		return coefficients;
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
