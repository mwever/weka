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

package weka.filters.supervised.attribute;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Vector;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.ContingencyTables;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.RevisionUtils;
import weka.core.SparseInstance;
import weka.core.SpecialFunctions;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedAttributesHandler;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

/**
 * <!-- globalinfo-start --> An instance filter that discretizes a range of numeric attributes in
 * the dataset into nominal attributes. Discretization is by Fayyad &amp; Irani's MDL method (the
 * default).<br/>
 * <br/>
 * For more information, see:<br/>
 * <br/>
 * Usama M. Fayyad, Keki B. Irani: Multi-interval discretization of continuousvalued attributes for
 * classification learning. In: Thirteenth International Joint Conference on Articial Intelligence,
 * 1022-1027, 1993.<br/>
 * <br/>
 * Igor Kononenko: On Biases in Estimating Multi-Valued Attributes. In: 14th International Joint
 * Conference on Articial Intelligence, 1034-1040, 1995.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#64;inproceedings{Fayyad1993,
 *    author = {Usama M. Fayyad and Keki B. Irani},
 *    booktitle = {Thirteenth International Joint Conference on Articial Intelligence},
 *    pages = {1022-1027},
 *    publisher = {Morgan Kaufmann Publishers},
 *    title = {Multi-interval discretization of continuousvalued attributes for classification learning},
 *    volume = {2},
 *    year = {1993}
 * }
 *
 * &#64;inproceedings{Kononenko1995,
 *    author = {Igor Kononenko},
 *    booktitle = {14th International Joint Conference on Articial Intelligence},
 *    pages = {1034-1040},
 *    title = {On Biases in Estimating Multi-Valued Attributes},
 *    year = {1995},
 *    PS = {http://ai.fri.uni-lj.si/papers/kononenko95-ijcai.ps.gz}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 *  -R &lt;col1,col2-col4,...&gt;
 *  Specifies list of columns to Discretize. First and last are valid indexes.
 *  (default none)
 * </pre>
 *
 * <pre>
 *  -V
 *  Invert matching sense of column indexes.
 * </pre>
 *
 * <pre>
 *  -D
 *  Output binary attributes for discretized attributes.
 * </pre>
 *
 * <pre>
 *  -Y
 *  Use bin numbers rather than ranges for discretized attributes.
 * </pre>
 *
 * <pre>
 *  -E
 *  Use better encoding of split point for MDL.
 * </pre>
 *
 * <pre>
 *  -K
 *  Use Kononenko's MDL criterion.
 * </pre>
 *
 * <pre>
 *  -precision &lt;integer&gt;
 *  Precision for bin boundary labels.
 *  (default = 6 decimal places).
 * </pre>
 *
 * <pre>
 * -spread-attribute-weight
 *  When generating binary attributes, spread weight of old
 *  attribute across new attributes. Do not give each new attribute the old weight.
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class Discretize extends Filter implements SupervisedFilter, OptionHandler, WeightedInstancesHandler, WeightedAttributesHandler, TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = -3141006402280129097L;

	/** Stores which columns to Discretize */
	protected Range m_DiscretizeCols = new Range();

	/** Store the current cutpoints */
	protected double[][] m_CutPoints = null;

	/** Output binary attributes for discretized attributes. */
	protected boolean m_MakeBinary = false;

	/** Use bin numbers rather than ranges for discretized attributes. */
	protected boolean m_UseBinNumbers = false;

	/** Use better encoding of split point for MDL. */
	protected boolean m_UseBetterEncoding = false;

	/** Use Kononenko's MDL criterion instead of Fayyad et al.'s */
	protected boolean m_UseKononenko = false;

	/** Precision for bin range labels */
	protected int m_BinRangePrecision = 6;

	/** Whether to spread attribute weight when creating binary attributes */
	protected boolean m_SpreadAttributeWeight = false;

	/** Constructor - initialises the filter */
	public Discretize() {

		this.setAttributeIndices("first-last");
	}

	/**
	 * Gets an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<>(6);

		newVector.addElement(new Option("\tSpecifies list of columns to Discretize. First" + " and last are valid indexes.\n" + "\t(default none)", "R", 1, "-R <col1,col2-col4,...>"));

		newVector.addElement(new Option("\tInvert matching sense of column indexes.", "V", 0, "-V"));

		newVector.addElement(new Option("\tOutput binary attributes for discretized attributes.", "D", 0, "-D"));

		newVector.addElement(new Option("\tUse bin numbers rather than ranges for discretized attributes.", "Y", 0, "-Y"));

		newVector.addElement(new Option("\tUse better encoding of split point for MDL.", "E", 0, "-E"));

		newVector.addElement(new Option("\tUse Kononenko's MDL criterion.", "K", 0, "-K"));

		newVector.addElement(new Option("\tPrecision for bin boundary labels.\n\t" + "(default = 6 decimal places).", "precision", 1, "-precision <integer>"));

		newVector.addElement(
				new Option("\tWhen generating binary attributes, spread weight of old " + "attribute across new attributes. Do not give each new attribute the old weight.\n\t", "spread-attribute-weight", 0, "-spread-attribute-weight"));

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
	 *  -R &lt;col1,col2-col4,...&gt;
	 *  Specifies list of columns to Discretize. First and last are valid indexes.
	 *  (default none)
	 * </pre>
	 *
	 * <pre>
	 *  -V
	 *  Invert matching sense of column indexes.
	 * </pre>
	 *
	 * <pre>
	 *  -D
	 *  Output binary attributes for discretized attributes.
	 * </pre>
	 *
	 * <pre>
	 *  -Y
	 *  Use bin numbers rather than ranges for discretized attributes.
	 * </pre>
	 *
	 * <pre>
	 *  -E
	 *  Use better encoding of split point for MDL.
	 * </pre>
	 *
	 * <pre>
	 *  -K
	 *  Use Kononenko's MDL criterion.
	 * </pre>
	 *
	 * <pre>
	 *  -precision &lt;integer&gt;
	 *  Precision for bin boundary labels.
	 *  (default = 6 decimal places).
	 * </pre>
	 *
	 * <pre>
	 * -spread-attribute-weight
	 *  When generating binary attributes, spread weight of old
	 *  attribute across new attributes. Do not give each new attribute the old weight.
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

		this.setMakeBinary(Utils.getFlag('D', options));
		this.setUseBinNumbers(Utils.getFlag('Y', options));
		this.setUseBetterEncoding(Utils.getFlag('E', options));
		this.setUseKononenko(Utils.getFlag('K', options));
		this.setInvertSelection(Utils.getFlag('V', options));

		String convertList = Utils.getOption('R', options);
		if (convertList.length() != 0) {
			this.setAttributeIndices(convertList);
		} else {
			this.setAttributeIndices("first-last");
		}

		String precisionS = Utils.getOption("precision", options);
		if (precisionS.length() > 0) {
			this.setBinRangePrecision(Integer.parseInt(precisionS));
		}

		this.setSpreadAttributeWeight(Utils.getFlag("spread-attribute-weight", options));

		if (this.getInputFormat() != null) {
			this.setInputFormat(this.getInputFormat());
		}

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of the filter.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {

		List<String> options = new ArrayList<>();

		if (this.getMakeBinary()) {
			options.add("-D");
		}
		if (this.getUseBinNumbers()) {
			options.add("-Y");
		}
		if (this.getUseBetterEncoding()) {
			options.add("-E");
		}
		if (this.getUseKononenko()) {
			options.add("-K");
		}
		if (this.getInvertSelection()) {
			options.add("-V");
		}
		if (!this.getAttributeIndices().equals("")) {
			options.add("-R");
			options.add(this.getAttributeIndices());
		}

		options.add("-precision");
		options.add("" + this.getBinRangePrecision());

		if (this.getSpreadAttributeWeight()) {
			options.add("-spread-attribute-weight");
		}

		return options.toArray(new String[options.size()]);
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
		result.enable(Capability.NOMINAL_CLASS);

		return result;
	}

	/**
	 * Sets the format of the input instances.
	 *
	 * @param instanceInfo
	 *          an Instances object containing the input instance structure (any instances contained in
	 *          the object are ignored - only the structure is required).
	 * @return true if the outputFormat may be collected immediately
	 * @throws Exception
	 *           if the input format can't be set successfully
	 */
	@Override
	public boolean setInputFormat(final Instances instanceInfo) throws Exception {

		super.setInputFormat(instanceInfo);

		this.m_DiscretizeCols.setUpper(instanceInfo.numAttributes() - 1);
		this.m_CutPoints = null;

		// If we implement loading cutfiles, then load
		// them here and set the output format
		return false;
	}

	/**
	 * Input an instance for filtering. Ordinarily the instance is processed and made available for
	 * output immediately. Some filters require all instances be read before producing output.
	 *
	 * @param instance
	 *          the input instance
	 * @return true if the filtered instance may now be collected with output().
	 * @throws InterruptedException
	 * @throws IllegalStateException
	 *           if no input format has been defined.
	 */
	@Override
	public boolean input(final Instance instance) throws InterruptedException {

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
	 * Signifies that this batch of input to the filter is finished. If the filter requires all
	 * instances prior to filtering, output() may now be called to retrieve the filtered instances.
	 *
	 * @return true if there are instances pending output
	 * @throws InterruptedException
	 * @throws IllegalStateException
	 *           if no input structure has been defined
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
	 * @return a description of the filter suitable for displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return "An instance filter that discretizes a range of numeric" + " attributes in the dataset into nominal attributes." + " Discretization is by Fayyad & Irani's MDL method (the default).\n\n" + "For more information, see:\n\n"
				+ this.getTechnicalInformation().toString();
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
		TechnicalInformation additional;

		result = new TechnicalInformation(Type.INPROCEEDINGS);
		result.setValue(Field.AUTHOR, "Usama M. Fayyad and Keki B. Irani");
		result.setValue(Field.TITLE, "Multi-interval discretization of continuousvalued attributes for classification learning");
		result.setValue(Field.BOOKTITLE, "Thirteenth International Joint Conference on Articial Intelligence");
		result.setValue(Field.YEAR, "1993");
		result.setValue(Field.VOLUME, "2");
		result.setValue(Field.PAGES, "1022-1027");
		result.setValue(Field.PUBLISHER, "Morgan Kaufmann Publishers");

		additional = result.add(Type.INPROCEEDINGS);
		additional.setValue(Field.AUTHOR, "Igor Kononenko");
		additional.setValue(Field.TITLE, "On Biases in Estimating Multi-Valued Attributes");
		additional.setValue(Field.BOOKTITLE, "14th International Joint Conference on Articial Intelligence");
		additional.setValue(Field.YEAR, "1995");
		additional.setValue(Field.PAGES, "1034-1040");
		additional.setValue(Field.PS, "http://ai.fri.uni-lj.si/papers/kononenko95-ijcai.ps.gz");

		return result;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String spreadAttributeWeightTipText() {
		return "When generating binary attributes, spread weight of old attribute across new attributes. " + "Do not give each new attribute the old weight.";
	}

	/**
	 * If true, when generating binary attributes, spread weight of old attribute across new attributes.
	 * Do not give each new attribute the old weight.
	 *
	 * @param p
	 *          whether weight is spread
	 */
	public void setSpreadAttributeWeight(final boolean p) {
		this.m_SpreadAttributeWeight = p;
	}

	/**
	 * If true, when generating binary attributes, spread weight of old attribute across new attributes.
	 * Do not give each new attribute the old weight.
	 *
	 * @return whether weight is spread
	 */
	public boolean getSpreadAttributeWeight() {
		return this.m_SpreadAttributeWeight;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String binRangePrecisionTipText() {
		return "The number of decimal places for cut points to use when generating bin labels";
	}

	/**
	 * Set the precision for bin boundaries. Only affects the boundary values used in the labels for the
	 * converted attributes; internal cutpoints are at full double precision.
	 *
	 * @param p
	 *          the precision for bin boundaries
	 */
	public void setBinRangePrecision(final int p) {
		this.m_BinRangePrecision = p;
	}

	/**
	 * Get the precision for bin boundaries. Only affects the boundary values used in the labels for the
	 * converted attributes; internal cutpoints are at full double precision.
	 *
	 * @return the precision for bin boundaries
	 */
	public int getBinRangePrecision() {
		return this.m_BinRangePrecision;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
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
	 * @param makeBinary
	 *          if binary attributes are to be made
	 */
	public void setMakeBinary(final boolean makeBinary) {

		this.m_MakeBinary = makeBinary;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String useBinNumbersTipText() {
		return "Use bin numbers (eg BXofY) rather than ranges fordiscretized attributes";
	}

	/**
	 * Gets whether bin numbers rather than ranges should be used for discretized attributes.
	 *
	 * @return true if bin numbers should be used
	 */
	public boolean getUseBinNumbers() {

		return this.m_UseBinNumbers;
	}

	/**
	 * Sets whether bin numbers rather than ranges should be used for discretized attributes.
	 *
	 * @param useBinNumbers
	 *          if bin numbers should be used
	 */
	public void setUseBinNumbers(final boolean useBinNumbers) {

		this.m_UseBinNumbers = useBinNumbers;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String useKononenkoTipText() {

		return "Use Kononenko's MDL criterion. If set to false" + " uses the Fayyad & Irani criterion.";
	}

	/**
	 * Gets whether Kononenko's MDL criterion is to be used.
	 *
	 * @return true if Kononenko's criterion will be used.
	 */
	public boolean getUseKononenko() {

		return this.m_UseKononenko;
	}

	/**
	 * Sets whether Kononenko's MDL criterion is to be used.
	 *
	 * @param useKon
	 *          true if Kononenko's one is to be used
	 */
	public void setUseKononenko(final boolean useKon) {

		this.m_UseKononenko = useKon;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String useBetterEncodingTipText() {

		return "Uses a more efficient split point encoding.";
	}

	/**
	 * Gets whether better encoding is to be used for MDL.
	 *
	 * @return true if the better MDL encoding will be used
	 */
	public boolean getUseBetterEncoding() {

		return this.m_UseBetterEncoding;
	}

	/**
	 * Sets whether better encoding is to be used for MDL.
	 *
	 * @param useBetterEncoding
	 *          true if better encoding to be used.
	 */
	public void setUseBetterEncoding(final boolean useBetterEncoding) {

		this.m_UseBetterEncoding = useBetterEncoding;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
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
	 * Sets whether selected columns should be removed or kept. If true the selected columns are kept
	 * and unselected columns are deleted. If false selected columns are deleted and unselected columns
	 * are kept.
	 *
	 * @param invert
	 *          the new invert setting
	 */
	public void setInvertSelection(final boolean invert) {

		this.m_DiscretizeCols.setInvert(invert);
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
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
	 * Sets which attributes are to be Discretized (only numeric attributes among the selection will be
	 * Discretized).
	 *
	 * @param rangeList
	 *          a string representing the list of attributes. Since the string will typically come from
	 *          a user, attributes are indexed from 1. <br>
	 *          eg: first-3,5,6-last
	 * @throws IllegalArgumentException
	 *           if an invalid range list is supplied
	 */
	public void setAttributeIndices(final String rangeList) {

		this.m_DiscretizeCols.setRanges(rangeList);
	}

	/**
	 * Sets which attributes are to be Discretized (only numeric attributes among the selection will be
	 * Discretized).
	 *
	 * @param attributes
	 *          an array containing indexes of attributes to Discretize. Since the array will typically
	 *          come from a program, attributes are indexed from 0.
	 * @throws IllegalArgumentException
	 *           if an invalid set of ranges is supplied
	 */
	public void setAttributeIndicesArray(final int[] attributes) {

		this.setAttributeIndices(Range.indicesToRangeList(attributes));
	}

	/**
	 * Gets the cut points for an attribute
	 *
	 * @param attributeIndex
	 *          the index (from 0) of the attribute to get the cut points of
	 * @return an array containing the cutpoints (or null if the attribute requested isn't being
	 *         Discretized
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
	 * @param attributeIndex
	 *          the index (from 0) of the attribute to get the bin ranges string of
	 * @return the bin ranges string (or null if the attribute requested has been discretized into only
	 *         one interval.)
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

			sb.append(binRangeString(cutPoints, j, this.m_BinRangePrecision));
		}

		return sb.toString();
	}

	/**
	 * Get a bin range string for a specified bin of some attribute's cut points.
	 *
	 * @param cutPoints
	 *          The attribute's cut points; never null.
	 * @param j
	 *          The bin number (zero based); never out of range.
	 * @param precision
	 *          the precision for the range values
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

	/**
	 * Generate the cutpoints for each attribute
	 *
	 * @throws InterruptedException
	 */
	protected void calculateCutPoints() throws InterruptedException {

		Instances copy = null;

		this.m_CutPoints = new double[this.getInputFormat().numAttributes()][];
		for (int i = this.getInputFormat().numAttributes() - 1; i >= 0; i--) {
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}

			if ((this.m_DiscretizeCols.isInRange(i)) && (this.getInputFormat().attribute(i).isNumeric())) {

				// Use copy to preserve order
				if (copy == null) {
					copy = new Instances(this.getInputFormat());
				}
				this.calculateCutPointsByMDL(i, copy);
			}
		}
	}

	/**
	 * Set cutpoints for a single attribute using MDL.
	 *
	 * @param index
	 *          the index of the attribute to set cutpoints for
	 * @param data
	 *          the data to work with
	 * @throws InterruptedException
	 */
	protected void calculateCutPointsByMDL(final int index, final Instances data) throws InterruptedException {

		// Sort instances
		data.sort(data.attribute(index));

		// Find first instances that's missing
		int firstMissing = data.numInstances();
		for (int i = 0; i < data.numInstances(); i++) {
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			if (data.instance(i).isMissing(index)) {
				firstMissing = i;
				break;
			}
		}
		this.m_CutPoints[index] = this.cutPointsForSubset(data, index, 0, firstMissing);
	}

	/**
	 * Test using Kononenko's MDL criterion.
	 *
	 * @param priorCounts
	 * @param bestCounts
	 * @param numInstances
	 * @param numCutPoints
	 * @return true if the split is acceptable
	 */
	private boolean KononenkosMDL(final double[] priorCounts, final double[][] bestCounts, final double numInstances, final int numCutPoints) {

		double distPrior, instPrior, distAfter = 0, sum, instAfter = 0;
		double before, after;
		int numClassesTotal;

		// Number of classes occuring in the set
		numClassesTotal = 0;
		for (double priorCount : priorCounts) {
			if (priorCount > 0) {
				numClassesTotal++;
			}
		}

		// Encode distribution prior to split
		distPrior = SpecialFunctions.log2Binomial(numInstances + numClassesTotal - 1, numClassesTotal - 1);

		// Encode instances prior to split.
		instPrior = SpecialFunctions.log2Multinomial(numInstances, priorCounts);

		before = instPrior + distPrior;

		// Encode distributions and instances after split.
		for (double[] bestCount : bestCounts) {
			sum = Utils.sum(bestCount);
			distAfter += SpecialFunctions.log2Binomial(sum + numClassesTotal - 1, numClassesTotal - 1);
			instAfter += SpecialFunctions.log2Multinomial(sum, bestCount);
		}

		// Coding cost after split
		after = Utils.log2(numCutPoints) + distAfter + instAfter;

		// Check if split is to be accepted
		return (before > after);
	}

	/**
	 * Test using Fayyad and Irani's MDL criterion.
	 *
	 * @param priorCounts
	 * @param bestCounts
	 * @param numInstances
	 * @param numCutPoints
	 * @return true if the splits is acceptable
	 */
	private boolean FayyadAndIranisMDL(final double[] priorCounts, final double[][] bestCounts, final double numInstances, final int numCutPoints) {

		double priorEntropy, entropy, gain;
		double entropyLeft, entropyRight, delta;
		int numClassesTotal, numClassesRight, numClassesLeft;

		// Compute entropy before split.
		priorEntropy = ContingencyTables.entropy(priorCounts);

		// Compute entropy after split.
		entropy = ContingencyTables.entropyConditionedOnRows(bestCounts);

		// Compute information gain.
		gain = priorEntropy - entropy;

		// Number of classes occuring in the set
		numClassesTotal = 0;
		for (double priorCount : priorCounts) {
			if (priorCount > 0) {
				numClassesTotal++;
			}
		}

		// Number of classes occuring in the left subset
		numClassesLeft = 0;
		for (int i = 0; i < bestCounts[0].length; i++) {
			if (bestCounts[0][i] > 0) {
				numClassesLeft++;
			}
		}

		// Number of classes occuring in the right subset
		numClassesRight = 0;
		for (int i = 0; i < bestCounts[1].length; i++) {
			if (bestCounts[1][i] > 0) {
				numClassesRight++;
			}
		}

		// Entropy of the left and the right subsets
		entropyLeft = ContingencyTables.entropy(bestCounts[0]);
		entropyRight = ContingencyTables.entropy(bestCounts[1]);

		// Compute terms for MDL formula
		delta = Utils.log2(Math.pow(3, numClassesTotal) - 2) - ((numClassesTotal * priorEntropy) - (numClassesRight * entropyRight) - (numClassesLeft * entropyLeft));

		// Check if split is to be accepted
		return (gain > (Utils.log2(numCutPoints) + delta) / numInstances);
	}

	/**
	 * Selects cutpoints for sorted subset.
	 *
	 * @param instances
	 * @param attIndex
	 * @param first
	 * @param lastPlusOne
	 * @return
	 * @throws InterruptedException
	 */
	private double[] cutPointsForSubset(final Instances instances, final int attIndex, final int first, final int lastPlusOne) throws InterruptedException {

		double[][] counts, bestCounts;
		double[] priorCounts, left, right, cutPoints;
		double currentCutPoint = -Double.MAX_VALUE, bestCutPoint = -1, currentEntropy, bestEntropy, priorEntropy, gain;
		int bestIndex = -1, numCutPoints = 0;
		double numInstances = 0;

		// Compute number of instances in set
		if ((lastPlusOne - first) < 2) {
			return null;
		}

		// Compute class counts.
		counts = new double[2][instances.numClasses()];
		for (int i = first; i < lastPlusOne; i++) {
			numInstances += instances.instance(i).weight();
			counts[1][(int) instances.instance(i).classValue()] += instances.instance(i).weight();
		}

		// Save prior counts
		priorCounts = new double[instances.numClasses()];
		System.arraycopy(counts[1], 0, priorCounts, 0, instances.numClasses());

		// Entropy of the full set
		priorEntropy = ContingencyTables.entropy(priorCounts);
		bestEntropy = priorEntropy;

		// Find best entropy.
		bestCounts = new double[2][instances.numClasses()];
		for (int i = first; i < (lastPlusOne - 1); i++) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			counts[0][(int) instances.instance(i).classValue()] += instances.instance(i).weight();
			counts[1][(int) instances.instance(i).classValue()] -= instances.instance(i).weight();
			if (instances.instance(i).value(attIndex) < instances.instance(i + 1).value(attIndex)) {
				currentCutPoint = (instances.instance(i).value(attIndex) + instances.instance(i + 1).value(attIndex)) / 2.0;
				currentEntropy = ContingencyTables.entropyConditionedOnRows(counts);
				if (currentEntropy < bestEntropy) {
					bestCutPoint = currentCutPoint;
					bestEntropy = currentEntropy;
					bestIndex = i;
					System.arraycopy(counts[0], 0, bestCounts[0], 0, instances.numClasses());
					System.arraycopy(counts[1], 0, bestCounts[1], 0, instances.numClasses());
				}
				numCutPoints++;
			}
		}

		// Use worse encoding?
		if (!this.m_UseBetterEncoding) {
			numCutPoints = (lastPlusOne - first) - 1;
		}

		// Checks if gain is zero
		gain = priorEntropy - bestEntropy;
		if (gain <= 0) {
			return null;
		}

		// Check if split is to be accepted
		if ((this.m_UseKononenko && this.KononenkosMDL(priorCounts, bestCounts, numInstances, numCutPoints)) || (!this.m_UseKononenko && this.FayyadAndIranisMDL(priorCounts, bestCounts, numInstances, numCutPoints))) {

			// Select split points for the left and right subsets
			left = this.cutPointsForSubset(instances, attIndex, first, bestIndex + 1);
			right = this.cutPointsForSubset(instances, attIndex, bestIndex + 1, lastPlusOne);

			// Merge cutpoints and return them
			if ((left == null) && (right) == null) {
				cutPoints = new double[1];
				cutPoints[0] = bestCutPoint;
			} else if (right == null) {
				cutPoints = new double[left.length + 1];
				System.arraycopy(left, 0, cutPoints, 0, left.length);
				cutPoints[left.length] = bestCutPoint;
			} else if (left == null) {
				cutPoints = new double[1 + right.length];
				cutPoints[0] = bestCutPoint;
				System.arraycopy(right, 0, cutPoints, 1, right.length);
			} else {
				cutPoints = new double[left.length + right.length + 1];
				System.arraycopy(left, 0, cutPoints, 0, left.length);
				cutPoints[left.length] = bestCutPoint;
				System.arraycopy(right, 0, cutPoints, left.length + 1, right.length);
			}

			return cutPoints;
		} else {
			return null;
		}
	}

	/**
	 * Set the output format. Takes the currently defined cutpoints and m_InputFormat and calls
	 * setOutputFormat(Instances) appropriately.
	 */
	protected void setOutputFormat() {

		if (this.m_CutPoints == null) {
			this.setOutputFormat(null);
			return;
		}
		ArrayList<Attribute> attributes = new ArrayList<>(this.getInputFormat().numAttributes());
		int classIndex = this.getInputFormat().classIndex();
		for (int i = 0, m = this.getInputFormat().numAttributes(); i < m; ++i) {
			if ((this.m_DiscretizeCols.isInRange(i)) && (this.getInputFormat().attribute(i).isNumeric())) {

				Set<String> cutPointsCheck = new HashSet<>();
				double[] cutPoints = this.m_CutPoints[i];
				if (!this.m_MakeBinary) {
					ArrayList<String> attribValues;
					if (cutPoints == null) {
						attribValues = new ArrayList<>(1);
						attribValues.add("'All'");
					} else {
						attribValues = new ArrayList<>(cutPoints.length + 1);
						if (this.m_UseBinNumbers) {
							for (int j = 0, n = cutPoints.length; j <= n; ++j) {
								attribValues.add("'B" + (j + 1) + "of" + (n + 1) + "'");
							}
						} else {
							for (int j = 0, n = cutPoints.length; j <= n; ++j) {
								String newBinRangeString = binRangeString(cutPoints, j, this.m_BinRangePrecision);
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
						ArrayList<String> attribValues = new ArrayList<>(1);
						attribValues.add("'All'");
						Attribute newAtt = new Attribute(this.getInputFormat().attribute(i).name(), attribValues);
						newAtt.setWeight(this.getInputFormat().attribute(i).weight());
						attributes.add(newAtt);
					} else {
						if (i < this.getInputFormat().classIndex()) {
							classIndex += cutPoints.length - 1;
						}
						for (int j = 0, n = cutPoints.length; j < n; ++j) {
							ArrayList<String> attribValues = new ArrayList<>(2);
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
	 * Convert a single instance over. The converted instance is added to the end of the output queue.
	 *
	 * @param instance
	 *          the instance to convert
	 * @throws InterruptedException
	 */
	protected void convertInstance(final Instance instance) throws InterruptedException {

		int index = 0;
		double[] vals = new double[this.outputFormatPeek().numAttributes()];
		// Copy and convert the values
		for (int i = 0; i < this.getInputFormat().numAttributes(); i++) {
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			if (this.m_DiscretizeCols.isInRange(i) && this.getInputFormat().attribute(i).isNumeric()) {
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
	 * @param argv
	 *          should contain arguments to the filter: use -h for help
	 */
	public static void main(final String[] argv) {
		runFilter(new Discretize(), argv);
	}
}
