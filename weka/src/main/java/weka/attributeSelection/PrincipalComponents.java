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
 *    PrincipalComponents.java
 *    Copyright (C) 2000-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.attributeSelection;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;

import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.SymmDenseEVD;
import no.uib.cipr.matrix.UpperSymmDenseMatrix;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Center;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * <!-- globalinfo-start --> Performs a principal components analysis and
 * transformation of the data. Use in conjunction with a Ranker search.
 * Dimensionality reduction is accomplished by choosing enough eigenvectors to
 * account for some percentage of the variance in the original data---default
 * 0.95 (95%). Attribute noise can be filtered by transforming to the PC space,
 * eliminating some of the worst eigenvectors, and then transforming back to the
 * original space.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -C
 *  Center (rather than standardize) the
 *  data and compute PCA using the covariance (rather
 *   than the correlation) matrix.
 * </pre>
 *
 * <pre>
 * -R
 *  Retain enough PC attributes to account
 *  for this proportion of variance in the original data.
 *  (default = 0.95)
 * </pre>
 *
 * <pre>
 * -O
 *  Transform through the PC space and
 *  back to the original space.
 * </pre>
 *
 * <pre>
 * -A
 *  Maximum number of attributes to include in
 *  transformed attribute names. (-1 = include all)
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Mark Hall (mhall@cs.waikato.ac.nz)
 * @author Gabi Schmidberger (gabi@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class PrincipalComponents extends UnsupervisedAttributeEvaluator implements AttributeTransformer, OptionHandler {

	/** for serialization */
	private static final long serialVersionUID = -3675307197777734007L;

	/** The data to transform analyse/transform */
	private Instances m_trainInstances;

	/** Keep a copy for the class attribute (if set) */
	private Instances m_trainHeader;

	/** The header for the transformed data format */
	private Instances m_transformedFormat;

	/** The header for data transformed back to the original space */
	private Instances m_originalSpaceFormat;

	/** Data has a class set */
	private boolean m_hasClass;

	/** Class index */
	private int m_classIndex;

	/** Number of attributes */
	private int m_numAttribs;

	/** Number of instances */
	private int m_numInstances;

	/** Correlation/covariance matrix for the original data */
	private UpperSymmDenseMatrix m_correlation;

	private double[] m_means;
	private double[] m_stdDevs;

	/**
	 * If true, center (rather than standardize) the data and compute PCA from
	 * covariance (rather than correlation) matrix.
	 */
	private boolean m_center = false;

	/**
	 * Will hold the unordered linear transformations of the (normalized) original
	 * data
	 */
	private double[][] m_eigenvectors;

	/** Eigenvalues for the corresponding eigenvectors */
	private double[] m_eigenvalues = null;

	/** Sorted eigenvalues */
	private int[] m_sortedEigens;

	/** sum of the eigenvalues */
	private double m_sumOfEigenValues = 0.0;

	/** Filters for original data */
	private ReplaceMissingValues m_replaceMissingFilter;
	private NominalToBinary m_nominalToBinFilter;
	private Remove m_attributeFilter;
	private Center m_centerFilter;
	private Standardize m_standardizeFilter;

	/** The number of attributes in the pc transformed data */
	private int m_outputNumAtts = -1;

	/**
	 * the amount of variance to cover in the original data when retaining the
	 * best n PC's
	 */
	private double m_coverVariance = 0.95;

	/**
	 * transform the data through the pc space and back to the original space ?
	 */
	private boolean m_transBackToOriginal = false;

	/** maximum number of attributes in the transformed attribute name */
	private int m_maxAttrsInName = 5;

	/**
	 * holds the transposed eigenvectors for converting back to the original space
	 */
	private double[][] m_eTranspose;

	/**
	 * Returns a string describing this attribute transformer
	 *
	 * @return a description of the evaluator suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Performs a principal components analysis and transformation of " + "the data. Use in conjunction with a Ranker search. Dimensionality " + "reduction is accomplished by choosing enough eigenvectors to "
				+ "account for some percentage of the variance in the original data---" + "default 0.95 (95%). Attribute noise can be filtered by transforming " + "to the PC space, eliminating some of the worst eigenvectors, and "
				+ "then transforming back to the original space.";
	}

	/**
	 * Returns an enumeration describing the available options.
	 * <p>
	 *
	 * @return an enumeration of all the available options.
	 **/
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(4);

		newVector.addElement(new Option("\tCenter (rather than standardize) the" + "\n\tdata and compute PCA using the covariance (rather" + "\n\t than the correlation) matrix.", "C", 0, "-C"));

		newVector.addElement(new Option("\tRetain enough PC attributes to account " + "\n\tfor this proportion of variance in " + "the original data.\n" + "\t(default = 0.95)", "R", 1, "-R"));

		newVector.addElement(new Option("\tTransform through the PC space and " + "\n\tback to the original space.", "O", 0, "-O"));

		newVector.addElement(new Option("\tMaximum number of attributes to include in " + "\n\ttransformed attribute names. (-1 = include all)", "A", 1, "-A"));
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
	 * -C
	 *  Center (rather than standardize) the
	 *  data and compute PCA using the covariance (rather
	 *   than the correlation) matrix.
	 * </pre>
	 *
	 * <pre>
	 * -R
	 *  Retain enough PC attributes to account
	 *  for this proportion of variance in the original data.
	 *  (default = 0.95)
	 * </pre>
	 *
	 * <pre>
	 * -O
	 *  Transform through the PC space and
	 *  back to the original space.
	 * </pre>
	 *
	 * <pre>
	 * -A
	 *  Maximum number of attributes to include in
	 *  transformed attribute names. (-1 = include all)
	 * </pre>
	 *
	 * <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	@Override
	public void setOptions(final String[] options) throws Exception {
		this.resetOptions();
		String optionString;

		optionString = Utils.getOption('R', options);
		if (optionString.length() != 0) {
			Double temp;
			temp = Double.valueOf(optionString);
			this.setVarianceCovered(temp.doubleValue());
		}
		optionString = Utils.getOption('A', options);
		if (optionString.length() != 0) {
			this.setMaximumAttributeNames(Integer.parseInt(optionString));
		}

		this.setTransformBackToOriginal(Utils.getFlag('O', options));
		this.setCenterData(Utils.getFlag('C', options));
	}

	/**
	 * Reset to defaults
	 */
	private void resetOptions() {
		this.m_coverVariance = 0.95;
		this.m_sumOfEigenValues = 0.0;
		this.m_transBackToOriginal = false;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String centerDataTipText() {
		return "Center (rather than standardize) the data. PCA will " + "be computed from the covariance (rather than correlation) " + "matrix";
	}

	/**
	 * Set whether to center (rather than standardize) the data. If set to true
	 * then PCA is computed from the covariance rather than correlation matrix.
	 *
	 * @param center true if the data is to be centered rather than standardized
	 */
	public void setCenterData(final boolean center) {
		this.m_center = center;
	}

	/**
	 * Get whether to center (rather than standardize) the data. If true then PCA
	 * is computed from the covariance rather than correlation matrix.
	 *
	 * @return true if the data is to be centered rather than standardized.
	 */
	public boolean getCenterData() {
		return this.m_center;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String varianceCoveredTipText() {
		return "Retain enough PC attributes to account for this proportion of " + "variance.";
	}

	/**
	 * Sets the amount of variance to account for when retaining principal
	 * components
	 *
	 * @param vc the proportion of total variance to account for
	 */
	public void setVarianceCovered(final double vc) {
		this.m_coverVariance = vc;
	}

	/**
	 * Gets the proportion of total variance to account for when retaining
	 * principal components
	 *
	 * @return the proportion of variance to account for
	 */
	public double getVarianceCovered() {
		return this.m_coverVariance;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String maximumAttributeNamesTipText() {
		return "The maximum number of attributes to include in transformed attribute names.";
	}

	/**
	 * Sets maximum number of attributes to include in transformed attribute
	 * names.
	 *
	 * @param m the maximum number of attributes
	 */
	public void setMaximumAttributeNames(final int m) {
		this.m_maxAttrsInName = m;
	}

	/**
	 * Gets maximum number of attributes to include in transformed attribute
	 * names.
	 *
	 * @return the maximum number of attributes
	 */
	public int getMaximumAttributeNames() {
		return this.m_maxAttrsInName;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String transformBackToOriginalTipText() {
		return "Transform through the PC space and back to the original space. " + "If only the best n PCs are retained (by setting varianceCovered < 1) " + "then this option will give a dataset in the original space but with "
				+ "less attribute noise.";
	}

	/**
	 * Sets whether the data should be transformed back to the original space
	 *
	 * @param b true if the data should be transformed back to the original space
	 */
	public void setTransformBackToOriginal(final boolean b) {
		this.m_transBackToOriginal = b;
	}

	/**
	 * Gets whether the data is to be transformed back to the original space.
	 *
	 * @return true if the data is to be transformed back to the original space
	 */
	public boolean getTransformBackToOriginal() {
		return this.m_transBackToOriginal;
	}

	/**
	 * Gets the current settings of PrincipalComponents
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	@Override
	public String[] getOptions() {

		Vector<String> options = new Vector<String>();

		if (this.getCenterData()) {
			options.add("-C");
		}

		options.add("-R");
		options.add("" + this.getVarianceCovered());

		options.add("-A");
		options.add("" + this.getMaximumAttributeNames());

		if (this.getTransformBackToOriginal()) {
			options.add("-O");
		}

		return options.toArray(new String[0]);
	}

	/**
	 * Returns the capabilities of this evaluator.
	 *
	 * @return the capabilities of this evaluator
	 * @see Capabilities
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
		result.enable(Capability.UNARY_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.DATE_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);
		result.enable(Capability.NO_CLASS);

		return result;
	}

	/**
	 * Initializes principal components and performs the analysis
	 *
	 * @param data the instances to analyse/transform
	 * @throws Exception if analysis fails
	 */
	@Override
	public void buildEvaluator(final Instances data) throws Exception {
		// can evaluator handle data?
		this.getCapabilities().testWithFail(data);

		this.buildAttributeConstructor(data);
	}

	private void buildAttributeConstructor(final Instances data) throws Exception {
		this.m_eigenvalues = null;
		this.m_outputNumAtts = -1;
		this.m_attributeFilter = null;
		this.m_nominalToBinFilter = null;
		this.m_sumOfEigenValues = 0.0;
		this.m_trainInstances = new Instances(data);

		// make a copy of the training data so that we can get the class
		// column to append to the transformed data (if necessary)
		this.m_trainHeader = new Instances(this.m_trainInstances, 0);

		this.m_replaceMissingFilter = new ReplaceMissingValues();
		this.m_replaceMissingFilter.setInputFormat(this.m_trainInstances);
		this.m_trainInstances = Filter.useFilter(this.m_trainInstances, this.m_replaceMissingFilter);

		/*
		 * if (m_normalize) { m_normalizeFilter = new Normalize();
		 * m_normalizeFilter.setInputFormat(m_trainInstances); m_trainInstances =
		 * Filter.useFilter(m_trainInstances, m_normalizeFilter); }
		 */

		this.m_nominalToBinFilter = new NominalToBinary();
		this.m_nominalToBinFilter.setInputFormat(this.m_trainInstances);
		this.m_trainInstances = Filter.useFilter(this.m_trainInstances, this.m_nominalToBinFilter);

		// delete any attributes with only one distinct value or are all missing
		Vector<Integer> deleteCols = new Vector<Integer>();
		for (int i = 0; i < this.m_trainInstances.numAttributes(); i++) {
			// XXX thread interrupted; throw exception
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA");
			}
			if (this.m_trainInstances.numDistinctValues(i) <= 1) {
				deleteCols.addElement(new Integer(i));
			}
		}

		if (this.m_trainInstances.classIndex() >= 0) {
			// get rid of the class column
			this.m_hasClass = true;
			this.m_classIndex = this.m_trainInstances.classIndex();
			deleteCols.addElement(new Integer(this.m_classIndex));
		}

		// remove columns from the data if necessary
		if (deleteCols.size() > 0) {
			this.m_attributeFilter = new Remove();
			int[] todelete = new int[deleteCols.size()];
			for (int i = 0; i < deleteCols.size(); i++) {
				todelete[i] = (deleteCols.elementAt(i)).intValue();
			}
			this.m_attributeFilter.setAttributeIndicesArray(todelete);
			this.m_attributeFilter.setInvertSelection(false);
			this.m_attributeFilter.setInputFormat(this.m_trainInstances);
			this.m_trainInstances = Filter.useFilter(this.m_trainInstances, this.m_attributeFilter);
		}

		// can evaluator handle the processed data ? e.g., enough attributes?
		this.getCapabilities().testWithFail(this.m_trainInstances);

		this.m_numInstances = this.m_trainInstances.numInstances();
		this.m_numAttribs = this.m_trainInstances.numAttributes();

		this.fillCovariance();

		SymmDenseEVD evd = SymmDenseEVD.factorize(this.m_correlation);

		this.m_eigenvectors = Matrices.getArray(evd.getEigenvectors());
		this.m_eigenvalues = evd.getEigenvalues();

		/*
		 * for (int i = 0; i < m_numAttribs; i++) { for (int j = 0; j <
		 * m_numAttribs; j++) { System.err.println(v[i][j] + " "); }
		 * System.err.println(d[i]); }
		 */

		// any eigenvalues less than 0 are not worth anything --- change to 0
		for (int i = 0; i < this.m_eigenvalues.length; i++) {
			if (this.m_eigenvalues[i] < 0) {
				// XXX thread interrupted; throw exception
				if (Thread.interrupted()) {
					throw new InterruptedException("Killed WEKA");
				}
				this.m_eigenvalues[i] = 0.0;
			}
		}
		this.m_sortedEigens = Utils.sort(this.m_eigenvalues);
		this.m_sumOfEigenValues = Utils.sum(this.m_eigenvalues);

		this.m_transformedFormat = this.setOutputFormat();
		if (this.m_transBackToOriginal) {
			this.m_originalSpaceFormat = this.setOutputFormatOriginal();

			// new ordered eigenvector matrix
			int numVectors = (this.m_transformedFormat.classIndex() < 0) ? this.m_transformedFormat.numAttributes() : this.m_transformedFormat.numAttributes() - 1;

			double[][] orderedVectors = new double[this.m_eigenvectors.length][numVectors + 1];

			// try converting back to the original space
			for (int i = this.m_numAttribs - 1; i > (this.m_numAttribs - numVectors - 1); i--) {
				for (int j = 0; j < this.m_numAttribs; j++) {
					// XXX thread interrupted; throw exception
					if (Thread.interrupted()) {
						throw new InterruptedException("Killed WEKA");
					}
					orderedVectors[j][this.m_numAttribs - i] = this.m_eigenvectors[j][this.m_sortedEigens[i]];
				}
			}

			// transpose the matrix
			int nr = orderedVectors.length;
			int nc = orderedVectors[0].length;
			this.m_eTranspose = new double[nc][nr];
			for (int i = 0; i < nc; i++) {
				for (int j = 0; j < nr; j++) {
					// XXX thread interrupted; throw exception
					if (Thread.interrupted()) {
						throw new InterruptedException("Killed WEKA");
					}
					this.m_eTranspose[i][j] = orderedVectors[j][i];
				}
			}
		}
	}

	/**
	 * Returns just the header for the transformed data (ie. an empty set of
	 * instances. This is so that AttributeSelection can determine the structure
	 * of the transformed data without actually having to get all the transformed
	 * data through transformedData().
	 *
	 * @return the header of the transformed data.
	 * @throws Exception if the header of the transformed data can't be
	 *           determined.
	 */
	@Override
	public Instances transformedHeader() throws Exception {
		if (this.m_eigenvalues == null) {
			throw new Exception("Principal components hasn't been built yet");
		}
		if (this.m_transBackToOriginal) {
			return this.m_originalSpaceFormat;
		} else {
			return this.m_transformedFormat;
		}
	}

	/**
	 * Return the header of the training data after all filtering - i.e missing
	 * values and nominal to binary.
	 *
	 * @return the header of the training data after all filtering.
	 */
	public Instances getFilteredInputFormat() {
		return new Instances(this.m_trainInstances, 0);
	}

	/**
	 * Return the correlation/covariance matrix
	 *
	 * @return the correlation or covariance matrix
	 */
	public double[][] getCorrelationMatrix() {
		return Matrices.getArray(this.m_correlation);
	}

	/**
	 * Return the unsorted eigenvectors
	 *
	 * @return the unsorted eigenvectors
	 */
	public double[][] getUnsortedEigenVectors() {
		return this.m_eigenvectors;
	}

	/**
	 * Return the eigenvalues corresponding to the eigenvectors
	 *
	 * @return the eigenvalues
	 */
	public double[] getEigenValues() {
		return this.m_eigenvalues;
	}

	/**
	 * Gets the transformed training data.
	 *
	 * @return the transformed training data
	 * @throws Exception if transformed data can't be returned
	 */
	@Override
	public Instances transformedData(final Instances data) throws Exception {
		if (this.m_eigenvalues == null) {
			throw new Exception("Principal components hasn't been built yet");
		}

		Instances output = null;

		if (this.m_transBackToOriginal) {
			output = new Instances(this.m_originalSpaceFormat);
		} else {
			output = new Instances(this.m_transformedFormat);
		}
		for (int i = 0; i < data.numInstances(); i++) {
			// XXX thread interrupted; throw exception
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA");
			}
			Instance converted = this.convertInstance(data.instance(i));
			output.add(converted);
		}

		return output;
	}

	/**
	 * Evaluates the merit of a transformed attribute. This is defined to be 1
	 * minus the cumulative variance explained. Merit can't be meaningfully
	 * evaluated if the data is to be transformed back to the original space.
	 *
	 * @param att the attribute to be evaluated
	 * @return the merit of a transformed attribute
	 * @throws Exception if attribute can't be evaluated
	 */
	@Override
	public double evaluateAttribute(final int att) throws Exception {
		if (this.m_eigenvalues == null) {
			throw new Exception("Principal components hasn't been built yet!");
		}

		if (this.m_transBackToOriginal) {
			return 1.0; // can't evaluate back in the original space!
		}

		// return 1-cumulative variance explained for this transformed att
		double cumulative = 0.0;
		for (int i = this.m_numAttribs - 1; i >= this.m_numAttribs - att - 1; i--) {
			cumulative += this.m_eigenvalues[this.m_sortedEigens[i]];
		}

		return 1.0 - cumulative / this.m_sumOfEigenValues;
	}

	private void fillCovariance() throws Exception {
		// first store the means
		this.m_means = new double[this.m_trainInstances.numAttributes()];
		this.m_stdDevs = new double[this.m_trainInstances.numAttributes()];
		for (int i = 0; i < this.m_trainInstances.numAttributes(); i++) {
			// XXX thread interrupted; throw exception
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA");
			}
			this.m_means[i] = this.m_trainInstances.meanOrMode(i);
			this.m_stdDevs[i] = Math.sqrt(Utils.variance(this.m_trainInstances.attributeToDoubleArray(i)));
		}

		// just center the data or standardize it?
		if (this.m_center) {
			this.m_centerFilter = new Center();
			this.m_centerFilter.setInputFormat(this.m_trainInstances);
			this.m_trainInstances = Filter.useFilter(this.m_trainInstances, this.m_centerFilter);
		} else {
			this.m_standardizeFilter = new Standardize();
			this.m_standardizeFilter.setInputFormat(this.m_trainInstances);
			this.m_trainInstances = Filter.useFilter(this.m_trainInstances, this.m_standardizeFilter);
		}

		// now compute the covariance matrix
		this.m_correlation = new UpperSymmDenseMatrix(this.m_numAttribs);
		for (int i = 0; i < this.m_numAttribs; i++) {
			for (int j = i; j < this.m_numAttribs; j++) {
				// XXX thread interrupted; throw exception
				if (Thread.interrupted()) {
					throw new InterruptedException("Killed WEKA");
				}

				double cov = 0;
				for (Instance inst : this.m_trainInstances) {
					cov += inst.value(i) * inst.value(j);
				}

				cov /= this.m_trainInstances.numInstances() - 1;
				this.m_correlation.set(i, j, cov);
			}
		}
	}

	/**
	 * Return a summary of the analysis
	 *
	 * @return a summary of the analysis.
	 */
	private String principalComponentsSummary() {
		StringBuffer result = new StringBuffer();
		double cumulative = 0.0;
		Instances output = null;
		int numVectors = 0;

		try {
			output = this.setOutputFormat();
			numVectors = (output.classIndex() < 0) ? output.numAttributes() : output.numAttributes() - 1;
		} catch (Exception ex) {
		}
		// tomorrow
		String corrCov = (this.m_center) ? "Covariance " : "Correlation ";
		result.append(corrCov + "matrix\n" + matrixToString(Matrices.getArray(this.m_correlation)) + "\n\n");
		result.append("eigenvalue\tproportion\tcumulative\n");
		for (int i = this.m_numAttribs - 1; i > (this.m_numAttribs - numVectors - 1); i--) {
			cumulative += this.m_eigenvalues[this.m_sortedEigens[i]];
			result.append(Utils.doubleToString(this.m_eigenvalues[this.m_sortedEigens[i]], 9, 5) + "\t" + Utils.doubleToString((this.m_eigenvalues[this.m_sortedEigens[i]] / this.m_sumOfEigenValues), 9, 5) + "\t"
					+ Utils.doubleToString((cumulative / this.m_sumOfEigenValues), 9, 5) + "\t" + output.attribute(this.m_numAttribs - i - 1).name() + "\n");
		}

		result.append("\nEigenvectors\n");
		for (int j = 1; j <= numVectors; j++) {
			result.append(" V" + j + '\t');
		}
		result.append("\n");
		for (int j = 0; j < this.m_numAttribs; j++) {

			for (int i = this.m_numAttribs - 1; i > (this.m_numAttribs - numVectors - 1); i--) {
				result.append(Utils.doubleToString(this.m_eigenvectors[j][this.m_sortedEigens[i]], 7, 4) + "\t");
			}
			result.append(this.m_trainInstances.attribute(j).name() + '\n');
		}

		if (this.m_transBackToOriginal) {
			result.append("\nPC space transformed back to original space.\n" + "(Note: can't evaluate attributes in the original " + "space)\n");
		}
		return result.toString();
	}

	/**
	 * Returns a description of this attribute transformer
	 *
	 * @return a String describing this attribute transformer
	 */
	@Override
	public String toString() {
		if (this.m_eigenvalues == null) {
			return "Principal components hasn't been built yet!";
		} else {
			return "\tPrincipal Components Attribute Transformer\n\n" + this.principalComponentsSummary();
		}
	}

	/**
	 * Return a matrix as a String
	 *
	 * @param matrix that is decribed as a string
	 * @return a String describing a matrix
	 */
	public static String matrixToString(final double[][] matrix) {
		StringBuffer result = new StringBuffer();
		int last = matrix.length - 1;

		for (int i = 0; i <= last; i++) {
			for (int j = 0; j <= last; j++) {
				result.append(Utils.doubleToString(matrix[i][j], 6, 2) + " ");
				if (j == last) {
					result.append('\n');
				}
			}
		}
		return result.toString();
	}

	/**
	 * Convert a pc transformed instance back to the original space
	 *
	 * @param inst the instance to convert
	 * @return the processed instance
	 * @throws Exception if something goes wrong
	 */
	private Instance convertInstanceToOriginal(final Instance inst) throws Exception {
		double[] newVals = null;

		if (this.m_hasClass) {
			newVals = new double[this.m_numAttribs + 1];
		} else {
			newVals = new double[this.m_numAttribs];
		}

		if (this.m_hasClass) {
			// class is always appended as the last attribute
			newVals[this.m_numAttribs] = inst.value(inst.numAttributes() - 1);
		}

		for (int i = 0; i < this.m_eTranspose[0].length; i++) {
			double tempval = 0.0;
			for (int j = 1; j < this.m_eTranspose.length; j++) {
				tempval += (this.m_eTranspose[j][i] * inst.value(j - 1));
			}
			newVals[i] = tempval;
			if (!this.m_center) {
				newVals[i] *= this.m_stdDevs[i];
			}
			newVals[i] += this.m_means[i];
		}

		if (inst instanceof SparseInstance) {
			return new SparseInstance(inst.weight(), newVals);
		} else {
			return new DenseInstance(inst.weight(), newVals);
		}
	}

	/**
	 * Transform an instance in original (unormalized) format. Convert back to the
	 * original space if requested.
	 *
	 * @param instance an instance in the original (unormalized) format
	 * @return a transformed instance
	 * @throws Exception if instance cant be transformed
	 */
	@Override
	public Instance convertInstance(final Instance instance) throws Exception {

		if (this.m_eigenvalues == null) {
			throw new Exception("convertInstance: Principal components not " + "built yet");
		}

		double[] newVals = new double[this.m_outputNumAtts];
		Instance tempInst = (Instance) instance.copy();
		if (!instance.dataset().equalHeaders(this.m_trainHeader)) {
			throw new Exception("Can't convert instance: header's don't match: " + "PrincipalComponents\n" + instance.dataset().equalHeadersMsg(this.m_trainHeader));
		}

		this.m_replaceMissingFilter.input(tempInst);
		this.m_replaceMissingFilter.batchFinished();
		tempInst = this.m_replaceMissingFilter.output();

		/*
		 * if (m_normalize) { m_normalizeFilter.input(tempInst);
		 * m_normalizeFilter.batchFinished(); tempInst = m_normalizeFilter.output();
		 * }
		 */

		this.m_nominalToBinFilter.input(tempInst);
		this.m_nominalToBinFilter.batchFinished();
		tempInst = this.m_nominalToBinFilter.output();

		if (this.m_attributeFilter != null) {
			this.m_attributeFilter.input(tempInst);
			this.m_attributeFilter.batchFinished();
			tempInst = this.m_attributeFilter.output();
		}

		if (!this.m_center) {
			this.m_standardizeFilter.input(tempInst);
			this.m_standardizeFilter.batchFinished();
			tempInst = this.m_standardizeFilter.output();
		} else {
			this.m_centerFilter.input(tempInst);
			this.m_centerFilter.batchFinished();
			tempInst = this.m_centerFilter.output();
		}

		if (this.m_hasClass) {
			newVals[this.m_outputNumAtts - 1] = instance.value(instance.classIndex());
		}

		double cumulative = 0;
		for (int i = this.m_numAttribs - 1; i >= 0; i--) {
			// XXX thread interrupted; throw exception
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA");
			}
			double tempval = 0.0;
			for (int j = 0; j < this.m_numAttribs; j++) {
				// XXX thread interrupted; throw exception
				if (Thread.interrupted()) {
					throw new InterruptedException("Killed WEKA");
				}
				tempval += (this.m_eigenvectors[j][this.m_sortedEigens[i]] * tempInst.value(j));
			}
			newVals[this.m_numAttribs - i - 1] = tempval;
			cumulative += this.m_eigenvalues[this.m_sortedEigens[i]];
			if ((cumulative / this.m_sumOfEigenValues) >= this.m_coverVariance) {
				break;
			}
		}

		if (!this.m_transBackToOriginal) {
			if (instance instanceof SparseInstance) {
				return new SparseInstance(instance.weight(), newVals);
			} else {
				return new DenseInstance(instance.weight(), newVals);
			}
		} else {
			if (instance instanceof SparseInstance) {
				return this.convertInstanceToOriginal(new SparseInstance(instance.weight(), newVals));
			} else {
				return this.convertInstanceToOriginal(new DenseInstance(instance.weight(), newVals));
			}
		}
	}

	/**
	 * Set up the header for the PC->original space dataset
	 *
	 * @return the output format
	 * @throws Exception if something goes wrong
	 */
	private Instances setOutputFormatOriginal() throws Exception {
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();

		for (int i = 0; i < this.m_numAttribs; i++) {
			String att = this.m_trainInstances.attribute(i).name();
			attributes.add(new Attribute(att));
		}

		if (this.m_hasClass) {
			attributes.add((Attribute) this.m_trainHeader.classAttribute().copy());
		}

		Instances outputFormat = new Instances(this.m_trainHeader.relationName() + "->PC->original space", attributes, 0);

		// set the class to be the last attribute if necessary
		if (this.m_hasClass) {
			outputFormat.setClassIndex(outputFormat.numAttributes() - 1);
		}

		return outputFormat;
	}

	/**
	 * Set the format for the transformed data
	 *
	 * @return a set of empty Instances (header only) in the new format
	 * @throws Exception if the output format can't be set
	 */
	private Instances setOutputFormat() throws Exception {
		if (this.m_eigenvalues == null) {
			return null;
		}

		double cumulative = 0.0;
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = this.m_numAttribs - 1; i >= 0; i--) {
			// XXX thread interrupted; throw exception
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA");
			}
			StringBuffer attName = new StringBuffer();
			// build array of coefficients
			double[] coeff_mags = new double[this.m_numAttribs];
			for (int j = 0; j < this.m_numAttribs; j++) {
				coeff_mags[j] = -Math.abs(this.m_eigenvectors[j][this.m_sortedEigens[i]]);
			}
			int num_attrs = (this.m_maxAttrsInName > 0) ? Math.min(this.m_numAttribs, this.m_maxAttrsInName) : this.m_numAttribs;
			// this array contains the sorted indices of the coefficients
			int[] coeff_inds;
			if (this.m_numAttribs > 0) {
				// if m_maxAttrsInName > 0, sort coefficients by decreasing magnitude
				coeff_inds = Utils.sort(coeff_mags);
			} else {
				// if m_maxAttrsInName <= 0, use all coeffs in original order
				coeff_inds = new int[this.m_numAttribs];
				for (int j = 0; j < this.m_numAttribs; j++) {
					coeff_inds[j] = j;
				}
			}
			// build final attName string
			for (int j = 0; j < num_attrs; j++) {
				double coeff_value = this.m_eigenvectors[coeff_inds[j]][this.m_sortedEigens[i]];
				if (j > 0 && coeff_value >= 0) {
					attName.append("+");
				}
				attName.append(Utils.doubleToString(coeff_value, 5, 3) + this.m_trainInstances.attribute(coeff_inds[j]).name());
			}
			if (num_attrs < this.m_numAttribs) {
				attName.append("...");
			}

			attributes.add(new Attribute(attName.toString()));
			cumulative += this.m_eigenvalues[this.m_sortedEigens[i]];

			if ((cumulative / this.m_sumOfEigenValues) >= this.m_coverVariance) {
				break;
			}
		}

		if (this.m_hasClass) {
			attributes.add((Attribute) this.m_trainHeader.classAttribute().copy());
		}

		Instances outputFormat = new Instances(this.m_trainInstances.relationName() + "_principal components", attributes, 0);

		// set the class to be the last attribute if necessary
		if (this.m_hasClass) {
			outputFormat.setClassIndex(outputFormat.numAttributes() - 1);
		}

		this.m_outputNumAtts = outputFormat.numAttributes();
		return outputFormat;
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
	 * Main method for testing this class
	 *
	 * @param argv should contain the command line arguments to the
	 *          evaluator/transformer (see AttributeSelection)
	 */
	public static void main(final String[] argv) {
		runEvaluator(new PrincipalComponents(), argv);
	}
}
