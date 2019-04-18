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
 *    Discretize.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Set;
import java.util.Vector;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Range;
import weka.core.RevisionUtils;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.core.WeightedAttributesHandler;
import weka.core.WeightedInstancesHandler;
import weka.filters.UnsupervisedFilter;

/**
 * <!-- globalinfo-start --> An instance filter that discretizes a range of
 * numeric attributes in the dataset into nominal attributes. Discretization is
 * by simple binning. Skips the class attribute if set.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -unset-class-temporarily
 *  Unsets the class index temporarily before the filter is
 *  applied to the data.
 *  (default: no)
 * </pre>
 *
 * <pre>
 * -B &lt;num&gt;
 *  Specifies the (maximum) number of bins to divide numeric attributes into.
 *  (default = 10)
 * </pre>
 *
 * <pre>
 * -M &lt;num&gt;
 *  Specifies the desired weight of instances per bin for
 *  equal-frequency binning. If this is set to a positive
 *  number then the -B option will be ignored.
 *  (default = -1)
 * </pre>
 *
 * <pre>
 * -F
 *  Use equal-frequency instead of equal-width discretization.
 * </pre>
 *
 * <pre>
 * -O
 *  Optimize number of bins using leave-one-out estimate
 *  of estimated entropy (for equal-width discretization).
 *  If this is set then the -B option will be ignored.
 * </pre>
 *
 * <pre>
 * -R &lt;col1,col2-col4,...&gt;
 *  Specifies list of columns to Discretize. First and last are valid indexes.
 *  (default: first-last)
 * </pre>
 *
 * <pre>
 * -V
 *  Invert matching sense of column indexes.
 * </pre>
 *
 * <pre>
 * -D
 *  Output binary attributes for discretized attributes.
 * </pre>
 *
 * <pre>
 * -Y
 *  Use bin numbers rather than ranges for discretized attributes.
 * </pre>
 *
 * <pre> -precision &lt;integer&gt;
 *  Precision for bin boundary labels.
 *  (default = 6 decimal places).</pre>
 *
 * <pre>-spread-attribute-weight
 *  When generating binary attributes, spread weight of old
 *  attribute across new attributes. Do not give each new attribute the old weight.</pre>
 *
 * <!-- options-end -->
 *
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class Discretize extends PotentialClassIgnorer implements UnsupervisedFilter, WeightedInstancesHandler, WeightedAttributesHandler {

	/** for serialization */
	static final long serialVersionUID = -1358531742174527279L;

	/** Stores which columns to Discretize */
	protected Range m_DiscretizeCols = new Range();

	/** The number of bins to divide the attribute into */
	protected int m_NumBins = 10;

	/** The desired weight of instances per bin */
	protected double m_DesiredWeightOfInstancesPerInterval = -1;

	/** Store the current cutpoints */
	protected double[][] m_CutPoints = null;

	/** Output binary attributes for discretized attributes. */
	protected boolean m_MakeBinary = false;

	/** Use bin numbers rather than ranges for discretized attributes. */
	protected boolean m_UseBinNumbers = false;

	/** Find the number of bins using cross-validated entropy. */
	protected boolean m_FindNumBins = false;

	/** Use equal-frequency binning if unsupervised discretization turned on */
	protected boolean m_UseEqualFrequency = false;

	/** The default columns to discretize */
	protected String m_DefaultCols;

	/** Precision for bin range labels */
	protected int m_BinRangePrecision = 6;

	/** Whether to spread attribute weight when creating binary attributes */
	protected boolean m_SpreadAttributeWeight = false;

	/** Constructor - initialises the filter */
	public Discretize() {

		this.m_DefaultCols = "first-last";
		this.setAttributeIndices("first-last");
	}

	/**
	 * Another constructor, sets the attribute indices immediately
	 *
	 * @param cols the attribute indices
	 */
	public Discretize(final String cols) {

		this.m_DefaultCols = cols;
		this.setAttributeIndices(cols);
	}

	/**
	 * Gets an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> result = new Vector<Option>();

		result.addElement(new Option("\tSpecifies the (maximum) number of bins to divide numeric" + " attributes into.\n" + "\t(default = 10)", "B", 1, "-B <num>"));

		result.addElement(new Option("\tSpecifies the desired weight of instances per bin for\n" + "\tequal-frequency binning. If this is set to a positive\n" + "\tnumber then the -B option will be ignored.\n" + "\t(default = -1)", "M", 1,
				"-M <num>"));

		result.addElement(new Option("\tUse equal-frequency instead of equal-width discretization.", "F", 0, "-F"));

		result.addElement(new Option("\tOptimize number of bins using leave-one-out estimate\n" + "\tof estimated entropy (for equal-width discretization).\n" + "\tIf this is set then the -B option will be ignored.", "O", 0, "-O"));

		result.addElement(new Option("\tSpecifies list of columns to Discretize. First" + " and last are valid indexes.\n" + "\t(default: first-last)", "R", 1, "-R <col1,col2-col4,...>"));

		result.addElement(new Option("\tInvert matching sense of column indexes.", "V", 0, "-V"));

		result.addElement(new Option("\tOutput binary attributes for discretized attributes.", "D", 0, "-D"));

		result.addElement(new Option("\tUse bin numbers rather than ranges for discretized attributes.", "Y", 0, "-Y"));

		result.addElement(new Option("\tPrecision for bin boundary labels.\n\t" + "(default = 6 decimal places).", "precision", 1, "-precision <integer>"));

		result.addElement(
				new Option("\tWhen generating binary attributes, spread weight of old " + "attribute across new attributes. Do not give each new attribute the old weight.\n\t", "spread-attribute-weight", 0, "-spread-attribute-weight"));

		result.addAll(Collections.list(super.listOptions()));

		return result.elements();
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 *
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 *
	 * <pre>
	 * -unset-class-temporarily
	 *  Unsets the class index temporarily before the filter is
	 *  applied to the data.
	 *  (default: no)
	 * </pre>
	 *
	 * <pre>
	 * -B &lt;num&gt;
	 *  Specifies the (maximum) number of bins to divide numeric attributes into.
	 *  (default = 10)
	 * </pre>
	 *
	 * <pre>
	 * -M &lt;num&gt;
	 *  Specifies the desired weight of instances per bin for
	 *  equal-frequency binning. If this is set to a positive
	 *  number then the -B option will be ignored.
	 *  (default = -1)
	 * </pre>
	 *
	 * <pre>
	 * -F
	 *  Use equal-frequency instead of equal-width discretization.
	 * </pre>
	 *
	 * <pre>
	 * -O
	 *  Optimize number of bins using leave-one-out estimate
	 *  of estimated entropy (for equal-width discretization).
	 *  If this is set then the -B option will be ignored.
	 * </pre>
	 *
	 * <pre>
	 * -R &lt;col1,col2-col4,...&gt;
	 *  Specifies list of columns to Discretize. First and last are valid indexes.
	 *  (default: first-last)
	 * </pre>
	 *
	 * <pre>
	 * -V
	 *  Invert matching sense of column indexes.
	 * </pre>
	 *
	 * <pre>
	 * -D
	 *  Output binary attributes for discretized attributes.
	 * </pre>
	 *
	 * <pre>
	 * -Y
	 *  Use bin numbers rather than ranges for discretized attributes.
	 * </pre>
	 *
	 * <pre> -precision &lt;integer&gt;
	 *  Precision for bin boundary labels.
	 *  (default = 6 decimal places).</pre>
	 *
	 * <pre>-spread-attribute-weight
	 *  When generating binary attributes, spread weight of old
	 *  attribute across new attributes. Do not give each new attribute the old weight.</pre>
	 *
	 * <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	@Override
	public void setOptions(final String[] options) throws Exception {

		this.setMakeBinary(Utils.getFlag('D', options));
		this.setUseBinNumbers(Utils.getFlag('Y', options));
		this.setUseEqualFrequency(Utils.getFlag('F', options));
		this.setFindNumBins(Utils.getFlag('O', options));
		this.setInvertSelection(Utils.getFlag('V', options));

		String weight = Utils.getOption('M', options);
		if (weight.length() != 0) {
			this.setDesiredWeightOfInstancesPerInterval((new Double(weight)).doubleValue());
		} else {
			this.setDesiredWeightOfInstancesPerInterval(-1);
		}

		String numBins = Utils.getOption('B', options);
		if (numBins.length() != 0) {
			this.setBins(Integer.parseInt(numBins));
		} else {
			this.setBins(10);
		}

		String convertList = Utils.getOption('R', options);
		if (convertList.length() != 0) {
			this.setAttributeIndices(convertList);
		} else {
			this.setAttributeIndices(this.m_DefaultCols);
		}

		String precisionS = Utils.getOption("precision", options);
		if (precisionS.length() > 0) {
			this.setBinRangePrecision(Integer.parseInt(precisionS));
		}

		this.setSpreadAttributeWeight(Utils.getFlag("spread-attribute-weight", options));

		if (this.getInputFormat() != null) {
			this.setInputFormat(this.getInputFormat());
		}

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of the filter.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {

		Vector<String> result = new Vector<String>();

		if (this.getMakeBinary()) {
			result.add("-D");
		}

		if (this.getUseBinNumbers()) {
			result.add("-Y");
		}

		if (this.getUseEqualFrequency()) {
			result.add("-F");
		}

		if (this.getFindNumBins()) {
			result.add("-O");
		}

		if (this.getInvertSelection()) {
			result.add("-V");
		}

		result.add("-B");
		result.add("" + this.getBins());

		result.add("-M");
		result.add("" + this.getDesiredWeightOfInstancesPerInterval());

		if (!this.getAttributeIndices().equals("")) {
			result.add("-R");
			result.add(this.getAttributeIndices());
		}

		result.add("-precision");
		result.add("" + this.getBinRangePrecision());

		if (this.getSpreadAttributeWeight()) {
			result.add("-spread-attribute-weight");
		}

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Returns the Capabilities of this filter.
	 *
	 * @return the capabilities of this object
	 * @see Capabilities
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enableAllAttributes();
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enableAllClasses();
		result.enable(Capability.MISSING_CLASS_VALUES);
		if (!this.getMakeBinary()) {
			result.enable(Capability.NO_CLASS);
		}

		return result;
	}

	/**
	 * Sets the format of the input instances.
	 *
	 * @param instanceInfo an Instances object containing the input instance
	 *          structure (any instances contained in the object are ignored -
	 *          only the structure is required).
	 * @return true if the outputFormat may be collected immediately
	 * @throws Exception if the input format can't be set successfully
	 */
	@Override
	public boolean setInputFormat(final Instances instanceInfo) throws Exception {

		if (this.m_MakeBinary && this.m_IgnoreClass) {
			throw new IllegalArgumentException("Can't ignore class when " + "changing the number of attributes!");
		}

		super.setInputFormat(instanceInfo);

		this.m_DiscretizeCols.setUpper(instanceInfo.numAttributes() - 1);
		this.m_CutPoints = null;

		if (this.getFindNumBins() && this.getUseEqualFrequency()) {
			throw new IllegalArgumentException("Bin number optimization in conjunction " + "with equal-frequency binning not implemented.");
		}

		// If we implement loading cutfiles, then load
		// them here and set the output format
		return false;
	}

	/**
	 * Input an instance for filtering. Ordinarily the instance is processed and
	 * made available for output immediately. Some filters require all instances
	 * be read before producing output.
	 *
	 * @param instance the input instance
	 * @return true if the filtered instance may now be collected with output().
	 * @throws IllegalStateException if no input format has been defined.
	 */
	@Override
	public boolean input(final Instance instance) {

		if (this.getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}
		if (this.m_NewBatch) {
			this.resetQueue();
			this.m_NewBatch = false;
		}

		if (this.m_CutPoints != null) {
			this.convertInstance(instance);
			return true;
		}

		this.bufferInput(instance);
		return false;
	}

	/**
	 * Signifies that this batch of input to the filter is finished. If the filter
	 * requires all instances prior to filtering, output() may now be called to
	 * retrieve the filtered instances.
	 *
	 * @return true if there are instances pending output
	 * @throws InterruptedException
	 * @throws IllegalStateException if no input structure has been defined
	 */
	@Override
	public boolean batchFinished() throws InterruptedException {

		if (this.getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}
		if (this.m_CutPoints == null) {
			this.calculateCutPoints();

			this.setOutputFormat();

			// If we implement saving cutfiles, save the cuts here

			// Convert pending input instances
			for (int i = 0; i < this.getInputFormat().numInstances(); i++) {
				this.convertInstance(this.getInputFormat().instance(i));
			}
		}
		this.flushInput();

		this.m_NewBatch = true;
		return (this.numPendingOutput() != 0);
	}

	/**
	 * Returns a string describing this filter
	 *
	 * @return a description of the filter suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {

		return "An instance filter that discretizes a range of numeric" + " attributes in the dataset into nominal attributes." + " Discretization is by simple binning. Skips the class" + " attribute if set.";
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String findNumBinsTipText() {

		return "Optimize number of equal-width bins using leave-one-out. Doesn't " + "work for equal-frequency binning";
	}

	/**
	 * Get the value of FindNumBins.
	 *
	 * @return Value of FindNumBins.
	 */
	public boolean getFindNumBins() {

		return this.m_FindNumBins;
	}

	/**
	 * Set the value of FindNumBins.
	 *
	 * @param newFindNumBins Value to assign to FindNumBins.
	 */
	public void setFindNumBins(final boolean newFindNumBins) {

		this.m_FindNumBins = newFindNumBins;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String makeBinaryTipText() {

		return "Make resulting attributes binary.";
	}

	/**
	 * Gets whether binary attributes should be made for discretized ones.
	 *
	 * @return true if attributes will be binarized
	 */
	public boolean getMakeBinary() {

		return this.m_MakeBinary;
	}

	/**
	 * Sets whether binary attributes should be made for discretized ones.
	 *
	 * @param makeBinary if binary attributes are to be made
	 */
	public void setMakeBinary(final boolean makeBinary) {

		this.m_MakeBinary = makeBinary;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String useBinNumbersTipText() {
		return "Use bin numbers (eg BXofY) rather than ranges for for discretized attributes";
	}

	/**
	 * Gets whether bin numbers rather than ranges should be used for discretized
	 * attributes.
	 *
	 * @return true if bin numbers should be used
	 */
	public boolean getUseBinNumbers() {

		return this.m_UseBinNumbers;
	}

	/**
	 * Sets whether bin numbers rather than ranges should be used for discretized
	 * attributes.
	 *
	 * @param useBinNumbers if bin numbers should be used
	 */
	public void setUseBinNumbers(final boolean useBinNumbers) {

		this.m_UseBinNumbers = useBinNumbers;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String desiredWeightOfInstancesPerIntervalTipText() {

		return "Sets the desired weight of instances per interval for " + "equal-frequency binning.";
	}

	/**
	 * Get the DesiredWeightOfInstancesPerInterval value.
	 *
	 * @return the DesiredWeightOfInstancesPerInterval value.
	 */
	public double getDesiredWeightOfInstancesPerInterval() {

		return this.m_DesiredWeightOfInstancesPerInterval;
	}

	/**
	 * Set the DesiredWeightOfInstancesPerInterval value.
	 *
	 * @param newDesiredNumber The new DesiredNumber value.
	 */
	public void setDesiredWeightOfInstancesPerInterval(final double newDesiredNumber) {

		this.m_DesiredWeightOfInstancesPerInterval = newDesiredNumber;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String useEqualFrequencyTipText() {

		return "If set to true, equal-frequency binning will be used instead of" + " equal-width binning.";
	}

	/**
	 * Get the value of UseEqualFrequency.
	 *
	 * @return Value of UseEqualFrequency.
	 */
	public boolean getUseEqualFrequency() {

		return this.m_UseEqualFrequency;
	}

	/**
	 * Set the value of UseEqualFrequency.
	 *
	 * @param newUseEqualFrequency Value to assign to UseEqualFrequency.
	 */
	public void setUseEqualFrequency(final boolean newUseEqualFrequency) {

		this.m_UseEqualFrequency = newUseEqualFrequency;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String binsTipText() {

		return "Number of bins.";
	}

	/**
	 * Gets the number of bins numeric attributes will be divided into
	 *
	 * @return the number of bins.
	 */
	public int getBins() {

		return this.m_NumBins;
	}

	/**
	 * Sets the number of bins to divide each selected numeric attribute into
	 *
	 * @param numBins the number of bins
	 */
	public void setBins(final int numBins) {

		this.m_NumBins = numBins;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String invertSelectionTipText() {

		return "Set attribute selection mode. If false, only selected" + " (numeric) attributes in the range will be discretized; if" + " true, only non-selected attributes will be discretized.";
	}

	/**
	 * Gets whether the supplied columns are to be removed or kept
	 *
	 * @return true if the supplied columns will be kept
	 */
	public boolean getInvertSelection() {

		return this.m_DiscretizeCols.getInvert();
	}

	/**
	 * Sets whether selected columns should be removed or kept. If true the
	 * selected columns are kept and unselected columns are deleted. If false
	 * selected columns are deleted and unselected columns are kept.
	 *
	 * @param invert the new invert setting
	 */
	public void setInvertSelection(final boolean invert) {

		this.m_DiscretizeCols.setInvert(invert);
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String attributeIndicesTipText() {
		return "Specify range of attributes to act on." + " This is a comma separated list of attribute indices, with" + " \"first\" and \"last\" valid values. Specify an inclusive" + " range with \"-\". E.g: \"first-3,5,6-10,last\".";
	}

	/**
	 * Gets the current range selection
	 *
	 * @return a string containing a comma separated list of ranges
	 */
	public String getAttributeIndices() {

		return this.m_DiscretizeCols.getRanges();
	}

	/**
	 * Sets which attributes are to be Discretized (only numeric attributes among
	 * the selection will be Discretized).
	 *
	 * @param rangeList a string representing the list of attributes. Since the
	 *          string will typically come from a user, attributes are indexed
	 *          from 1. <br>
	 *          eg: first-3,5,6-last
	 * @throws IllegalArgumentException if an invalid range list is supplied
	 */
	public void setAttributeIndices(final String rangeList) {

		this.m_DiscretizeCols.setRanges(rangeList);
	}

	/**
	 * Sets which attributes are to be Discretized (only numeric attributes among
	 * the selection will be Discretized).
	 *
	 * @param attributes an array containing indexes of attributes to Discretize.
	 *          Since the array will typically come from a program, attributes are
	 *          indexed from 0.
	 * @throws IllegalArgumentException if an invalid set of ranges is supplied
	 */
	public void setAttributeIndicesArray(final int[] attributes) {

		this.setAttributeIndices(Range.indicesToRangeList(attributes));
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String spreadAttributeWeightTipText() {
		return "When generating binary attributes, spread weight of old attribute across new attributes. " + "Do not give each new attribute the old weight.";
	}

	/**
	 * If true, when generating binary attributes, spread weight of old
	 * attribute across new attributes. Do not give each new attribute the old weight.
	 *
	 * @param p whether weight is spread
	 */
	public void setSpreadAttributeWeight(final boolean p) {
		this.m_SpreadAttributeWeight = p;
	}

	/**
	 * If true, when generating binary attributes, spread weight of old
	 * attribute across new attributes. Do not give each new attribute the old weight.
	 *
	 * @return whether weight is spread
	 */
	public boolean getSpreadAttributeWeight() {
		return this.m_SpreadAttributeWeight;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String binRangePrecisionTipText() {
		return "The number of decimal places for cut points to use when generating bin labels";
	}

	/**
	 * Set the precision for bin boundaries. Only affects the boundary values used
	 * in the labels for the converted attributes; internal cutpoints are at full
	 * double precision.
	 *
	 * @param p the precision for bin boundaries
	 */
	public void setBinRangePrecision(final int p) {
		this.m_BinRangePrecision = p;
	}

	/**
	 * Get the precision for bin boundaries. Only affects the boundary values used
	 * in the labels for the converted attributes; internal cutpoints are at full
	 * double precision.
	 *
	 * @return the precision for bin boundaries
	 */
	public int getBinRangePrecision() {
		return this.m_BinRangePrecision;
	}

	/**
	 * Gets the cut points for an attribute
	 *
	 * @param attributeIndex the index (from 0) of the attribute to get the cut
	 *          points of
	 * @return an array containing the cutpoints (or null if the attribute
	 *         requested has been discretized into only one interval.)
	 */
	public double[] getCutPoints(final int attributeIndex) {

		if (this.m_CutPoints == null) {
			return null;
		}
		return this.m_CutPoints[attributeIndex];
	}

	/**
	 * Gets the bin ranges string for an attribute
	 *
	 * @param attributeIndex the index (from 0) of the attribute to get the bin
	 *          ranges string of
	 * @return the bin ranges string (or null if the attribute requested has been
	 *         discretized into only one interval.)
	 */
	public String getBinRangesString(final int attributeIndex) {

		if (this.m_CutPoints == null) {
			return null;
		}

		double[] cutPoints = this.m_CutPoints[attributeIndex];

		if (cutPoints == null) {
			return "All";
		}

		StringBuilder sb = new StringBuilder();
		boolean first = true;

		for (int j = 0, n = cutPoints.length; j <= n; ++j) {
			if (first) {
				first = false;
			} else {
				sb.append(',');
			}

			sb.append(binRangeString(cutPoints, j, this.getBinRangePrecision()));
		}

		return sb.toString();
	}

	/**
	 * Get a bin range string for a specified bin of some attribute's cut points.
	 *
	 * @param cutPoints The attribute's cut points; never null.
	 * @param j The bin number (zero based); never out of range.
	 * @param precision the precision for the range values
	 *
	 * @return The bin range string.
	 */
	private static String binRangeString(final double[] cutPoints, final int j, final int precision) {
		assert cutPoints != null;

		int n = cutPoints.length;
		assert 0 <= j && j <= n;

		return j == 0 ? "" + "(" + "-inf" + "-" + Utils.doubleToString(cutPoints[0], precision) + "]"
				: j == n ? "" + "(" + Utils.doubleToString(cutPoints[n - 1], precision) + "-" + "inf" + ")" : "" + "(" + Utils.doubleToString(cutPoints[j - 1], precision) + "-" + Utils.doubleToString(cutPoints[j], precision) + "]";
	}

	/** Generate the cutpoints for each attribute
	 * @throws InterruptedException */
	protected void calculateCutPoints() throws InterruptedException {

		this.m_CutPoints = new double[this.getInputFormat().numAttributes()][];
		for (int i = this.getInputFormat().numAttributes() - 1; i >= 0; i--) {
			if ((this.m_DiscretizeCols.isInRange(i)) && (this.getInputFormat().attribute(i).isNumeric()) && (this.getInputFormat().classIndex() != i)) {
				if (this.m_FindNumBins) {
					this.findNumBins(i);
				} else if (!this.m_UseEqualFrequency) {
					this.calculateCutPointsByEqualWidthBinning(i);
				} else {
					this.calculateCutPointsByEqualFrequencyBinning(i);
				}
			}
		}
	}

	/**
	 * Set cutpoints for a single attribute.
	 *
	 * @param index the index of the attribute to set cutpoints for
	 */
	protected void calculateCutPointsByEqualWidthBinning(final int index) {

		// Scan for max and min values
		double max = 0, min = 1, currentVal;
		Instance currentInstance;
		for (int i = 0; i < this.getInputFormat().numInstances(); i++) {
			currentInstance = this.getInputFormat().instance(i);
			if (!currentInstance.isMissing(index)) {
				currentVal = currentInstance.value(index);
				if (max < min) {
					max = min = currentVal;
				}
				if (currentVal > max) {
					max = currentVal;
				}
				if (currentVal < min) {
					min = currentVal;
				}
			}
		}
		double binWidth = (max - min) / this.m_NumBins;
		double[] cutPoints = null;
		if ((this.m_NumBins > 1) && (binWidth > 0)) {
			cutPoints = new double[this.m_NumBins - 1];
			for (int i = 1; i < this.m_NumBins; i++) {
				cutPoints[i - 1] = min + binWidth * i;
			}
		}
		this.m_CutPoints[index] = cutPoints;
	}

	/**
	 * Set cutpoints for a single attribute.
	 *
	 * @param index the index of the attribute to set cutpoints for
	* @throws InterruptedException
	 */
	protected void calculateCutPointsByEqualFrequencyBinning(final int index) throws InterruptedException {

		// Copy data so that it can be sorted
		Instances data = new Instances(this.getInputFormat());

		// Sort input data
		data.sort(index);

		// Compute weight of instances without missing values
		double sumOfWeights = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			if (data.instance(i).isMissing(index)) {
				break;
			} else {
				sumOfWeights += data.instance(i).weight();
			}
		}
		double freq;
		double[] cutPoints = new double[this.m_NumBins - 1];
		if (this.getDesiredWeightOfInstancesPerInterval() > 0) {
			freq = this.getDesiredWeightOfInstancesPerInterval();
			cutPoints = new double[(int) (sumOfWeights / freq)];
		} else {
			freq = sumOfWeights / this.m_NumBins;
			cutPoints = new double[this.m_NumBins - 1];
		}

		// Compute break points
		double counter = 0, last = 0;
		int cpindex = 0, lastIndex = -1;
		for (int i = 0; i < data.numInstances() - 1; i++) {

			// Stop if value missing
			if (data.instance(i).isMissing(index)) {
				break;
			}
			counter += data.instance(i).weight();
			sumOfWeights -= data.instance(i).weight();

			// Do we have a potential breakpoint?
			if (data.instance(i).value(index) < data.instance(i + 1).value(index)) {

				// Have we passed the ideal size?
				if (counter >= freq) {

					// Is this break point worse than the last one?
					if (((freq - last) < (counter - freq)) && (lastIndex != -1)) {
						cutPoints[cpindex] = (data.instance(lastIndex).value(index) + data.instance(lastIndex + 1).value(index)) / 2;
						counter -= last;
						last = counter;
						lastIndex = i;
					} else {
						cutPoints[cpindex] = (data.instance(i).value(index) + data.instance(i + 1).value(index)) / 2;
						counter = 0;
						last = 0;
						lastIndex = -1;
					}
					cpindex++;
					freq = (sumOfWeights + counter) / ((cutPoints.length + 1) - cpindex);
				} else {
					lastIndex = i;
					last = counter;
				}
			}
		}

		// Check whether there was another possibility for a cut point
		if ((cpindex < cutPoints.length) && (lastIndex != -1)) {
			cutPoints[cpindex] = (data.instance(lastIndex).value(index) + data.instance(lastIndex + 1).value(index)) / 2;
			cpindex++;
		}

		// Did we find any cutpoints?
		if (cpindex == 0) {
			this.m_CutPoints[index] = null;
		} else {
			double[] cp = new double[cpindex];
			for (int i = 0; i < cpindex; i++) {
				cp[i] = cutPoints[i];
			}
			this.m_CutPoints[index] = cp;
		}
	}

	/**
	 * Optimizes the number of bins using leave-one-out cross-validation.
	 *
	 * @param index the attribute index
	 * @throws InterruptedException
	 */
	protected void findNumBins(final int index) throws InterruptedException {

		double min = Double.MAX_VALUE, max = -Double.MAX_VALUE, binWidth = 0, entropy, bestEntropy = Double.MAX_VALUE, currentVal;
		double[] distribution;
		int bestNumBins = 1;
		Instance currentInstance;

		// Find minimum and maximum
		for (int i = 0; i < this.getInputFormat().numInstances(); i++) {
			currentInstance = this.getInputFormat().instance(i);
			if (!currentInstance.isMissing(index)) {
				currentVal = currentInstance.value(index);
				if (currentVal > max) {
					max = currentVal;
				}
				if (currentVal < min) {
					min = currentVal;
				}
			}
		}

		// Find best number of bins
		for (int i = 0; i < this.m_NumBins; i++) {
			distribution = new double[i + 1];
			binWidth = (max - min) / (i + 1);

			// Compute distribution
			for (int j = 0; j < this.getInputFormat().numInstances(); j++) {
				currentInstance = this.getInputFormat().instance(j);
				if (!currentInstance.isMissing(index)) {
					for (int k = 0; k < i + 1; k++) {
						if (currentInstance.value(index) <= (min + (((double) k + 1) * binWidth))) {
							distribution[k] += currentInstance.weight();
							break;
						}
					}
				}
			}

			// Compute cross-validated entropy
			entropy = 0;
			for (int k = 0; k < i + 1; k++) {
				if (distribution[k] < 2) {
					entropy = Double.MAX_VALUE;
					break;
				}
				entropy -= distribution[k] * Math.log((distribution[k] - 1) / binWidth);
			}

			// Best entropy so far?
			if (entropy < bestEntropy) {
				bestEntropy = entropy;
				bestNumBins = i + 1;
			}
		}

		// Compute cut points
		double[] cutPoints = null;
		if ((bestNumBins > 1) && (binWidth > 0)) {
			cutPoints = new double[bestNumBins - 1];
			for (int i = 1; i < bestNumBins; i++) {
				cutPoints[i - 1] = min + binWidth * i;
			}
		}
		this.m_CutPoints[index] = cutPoints;
	}

	/**
	 * Set the output format. Takes the currently defined cutpoints and
	 * m_InputFormat and calls setOutputFormat(Instances) appropriately.
	 */
	protected void setOutputFormat() {

		if (this.m_CutPoints == null) {
			this.setOutputFormat(null);
			return;
		}
		ArrayList<Attribute> attributes = new ArrayList<Attribute>(this.getInputFormat().numAttributes());
		int classIndex = this.getInputFormat().classIndex();
		for (int i = 0, m = this.getInputFormat().numAttributes(); i < m; ++i) {
			if ((this.m_DiscretizeCols.isInRange(i)) && (this.getInputFormat().attribute(i).isNumeric()) && (this.getInputFormat().classIndex() != i)) {

				Set<String> cutPointsCheck = new HashSet<String>();
				double[] cutPoints = this.m_CutPoints[i];
				if (!this.m_MakeBinary) {
					ArrayList<String> attribValues;
					if (cutPoints == null) {
						attribValues = new ArrayList<String>(1);
						attribValues.add("'All'");
					} else {
						attribValues = new ArrayList<String>(cutPoints.length + 1);
						if (this.m_UseBinNumbers) {
							for (int j = 0, n = cutPoints.length; j <= n; ++j) {
								attribValues.add("'B" + (j + 1) + "of" + (n + 1) + "'");
							}
						} else {
							for (int j = 0, n = cutPoints.length; j <= n; ++j) {
								String newBinRangeString = binRangeString(cutPoints, j, this.getBinRangePrecision());
								if (!cutPointsCheck.add(newBinRangeString)) {
									throw new IllegalArgumentException("A duplicate bin range was detected. Try increasing the bin range precision.");
								}
								attribValues.add("'" + newBinRangeString + "'");
							}
						}
					}
					Attribute newAtt = new Attribute(this.getInputFormat().attribute(i).name(), attribValues);
					newAtt.setWeight(this.getInputFormat().attribute(i).weight());
					attributes.add(newAtt);
				} else {
					if (cutPoints == null) {
						ArrayList<String> attribValues = new ArrayList<String>(1);
						attribValues.add("'All'");
						Attribute newAtt = new Attribute(this.getInputFormat().attribute(i).name(), attribValues);
						newAtt.setWeight(this.getInputFormat().attribute(i).weight());
						attributes.add(newAtt);
					} else {
						if (i < this.getInputFormat().classIndex()) {
							classIndex += cutPoints.length - 1;
						}
						for (int j = 0, n = cutPoints.length; j < n; ++j) {
							ArrayList<String> attribValues = new ArrayList<String>(2);
							if (this.m_UseBinNumbers) {
								attribValues.add("'B1of2'");
								attribValues.add("'B2of2'");
							} else {
								double[] binaryCutPoint = { cutPoints[j] };
								String newBinRangeString1 = binRangeString(binaryCutPoint, 0, this.m_BinRangePrecision);
								String newBinRangeString2 = binRangeString(binaryCutPoint, 1, this.m_BinRangePrecision);
								if (newBinRangeString1.equals(newBinRangeString2)) {
									throw new IllegalArgumentException("A duplicate bin range was detected. Try increasing the bin range precision.");
								}
								attribValues.add("'" + newBinRangeString1 + "'");
								attribValues.add("'" + newBinRangeString2 + "'");
							}
							Attribute newAtt = new Attribute(this.getInputFormat().attribute(i).name() + "_" + (j + 1), attribValues);
							if (this.getSpreadAttributeWeight()) {
								newAtt.setWeight(this.getInputFormat().attribute(i).weight() / cutPoints.length);
							} else {
								newAtt.setWeight(this.getInputFormat().attribute(i).weight());
							}
							attributes.add(newAtt);
						}
					}
				}
			} else {
				attributes.add((Attribute) this.getInputFormat().attribute(i).copy());
			}
		}
		Instances outputFormat = new Instances(this.getInputFormat().relationName(), attributes, 0);
		outputFormat.setClassIndex(classIndex);
		this.setOutputFormat(outputFormat);
	}

	/**
	 * Convert a single instance over. The converted instance is added to the end
	 * of the output queue.
	 *
	 * @param instance the instance to convert
	 */
	protected void convertInstance(final Instance instance) {

		int index = 0;
		double[] vals = new double[this.outputFormatPeek().numAttributes()];
		// Copy and convert the values
		for (int i = 0; i < this.getInputFormat().numAttributes(); i++) {
			if (this.m_DiscretizeCols.isInRange(i) && this.getInputFormat().attribute(i).isNumeric() && (this.getInputFormat().classIndex() != i)) {
				int j;
				double currentVal = instance.value(i);
				if (this.m_CutPoints[i] == null) {
					if (instance.isMissing(i)) {
						vals[index] = Utils.missingValue();
					} else {
						vals[index] = 0;
					}
					index++;
				} else {
					if (!this.m_MakeBinary) {
						if (instance.isMissing(i)) {
							vals[index] = Utils.missingValue();
						} else {
							for (j = 0; j < this.m_CutPoints[i].length; j++) {
								if (currentVal <= this.m_CutPoints[i][j]) {
									break;
								}
							}
							vals[index] = j;
						}
						index++;
					} else {
						for (j = 0; j < this.m_CutPoints[i].length; j++) {
							if (instance.isMissing(i)) {
								vals[index] = Utils.missingValue();
							} else if (currentVal <= this.m_CutPoints[i][j]) {
								vals[index] = 0;
							} else {
								vals[index] = 1;
							}
							index++;
						}
					}
				}
			} else {
				vals[index] = instance.value(i);
				index++;
			}
		}

		Instance inst = null;
		if (instance instanceof SparseInstance) {
			inst = new SparseInstance(instance.weight(), vals);
		} else {
			inst = new DenseInstance(instance.weight(), vals);
		}

		this.copyValues(inst, false, instance.dataset(), this.outputFormatPeek());

		this.push(inst); // No need to copy instance
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
	 * @param argv should contain arguments to the filter: use -h for help
	 */
	public static void main(final String[] argv) {
		runFilter(new Discretize(), argv);
	}
}
