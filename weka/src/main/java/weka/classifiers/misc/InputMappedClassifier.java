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
 *    InputMappedClassifier.java
 *    Copyright (C) 2010-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.misc;

import java.io.Serializable;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.Drawable;
import weka.core.Environment;
import weka.core.EnvironmentHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 * <!-- globalinfo-start --> Wrapper classifier that addresses incompatible training and test data by building a mapping between the training data that a classifier has been built with and the incoming test instances' structure. Model
 * attributes that are not found in the incoming instances receive missing values, so do incoming nominal attribute values that the classifier has not seen before. A new classifier can be trained or an existing one loaded from a file.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -I
 *  Ignore case when matching attribute names and nominal values.
 * </pre>
 *
 * <pre>
 * -M
 *  Suppress the output of the mapping report.
 * </pre>
 *
 * <pre>
 * -trim
 *  Trim white space from either end of names before matching.
 * </pre>
 *
 * <pre>
 * -L &lt;path to model to load&gt;
 *  Path to a model to load. If set, this model
 *  will be used for prediction and any base classifier
 *  specification will be ignored. Environment variables
 *  may be used in the path (e.g. ${HOME}/myModel.model)
 * </pre>
 *
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 *
 * <pre>
 * -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.rules.ZeroR)
 * </pre>
 *
 * <pre>
 * Options specific to classifier weka.classifiers.rules.ZeroR:
 * </pre>
 *
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision$
 *
 */
public class InputMappedClassifier extends SingleClassifierEnhancer implements Serializable, OptionHandler, Drawable, WeightedInstancesHandler, AdditionalMeasureProducer, EnvironmentHandler {

	/** For serialization */
	private static final long serialVersionUID = 4901630631723287761L;

	/** The path to the serialized model to use (if any) */
	protected String m_modelPath = "";

	/** The header of the last known set of incoming test instances */
	protected transient Instances m_inputHeader;

	/** The instances structure used to train the classifier with */
	protected Instances m_modelHeader;

	/** Handle any environment variables used in the model path */
	protected transient Environment m_env;

	/** Map from model attributes to incoming attributes */
	protected transient int[] m_attributeMap;

	protected transient int[] m_attributeStatus;

	/**
	 * For each model attribute, map from incoming nominal values to model nominal values
	 */
	protected transient int[][] m_nominalValueMap;

	/** Trim white space from both ends of attribute names and nominal values? */
	protected boolean m_trim = true;

	/** Ignore case when matching attribute names and nominal values? */
	protected boolean m_ignoreCase = true;

	/** Dont output mapping report if set to true */
	protected boolean m_suppressMappingReport = false;

	/**
	 * If true, then a call to buildClassifier() will not overwrite any test structure that has been recorded with the current training structure. This is useful for getting a correct mapping report output in toString() after
	 * buildClassifier has been called and before any test instance has been seen. Test structure and mapping will get reset if a test instance is received whose structure does not match the recorded test structure.
	 */
	protected boolean m_initialTestStructureKnown = false;

	/** Holds values for instances constructed for prediction */
	protected double[] m_vals;

	/**
	 * Returns a string describing this classifier
	 *
	 * @return a description of the classifier suitable for displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Wrapper classifier that addresses incompatible training and test " + "data by building a mapping between the training data that " + "a classifier has been built with and the incoming test instances' "
				+ "structure. Model attributes that are not found in the incoming " + "instances receive missing values, so do incoming nominal attribute " + "values that the classifier has not seen before. A new classifier "
				+ "can be trained or an existing one loaded from a file.";
	}

	/**
	 * Set the environment variables to use
	 *
	 * @param env
	 *            the environment variables to use
	 */
	@Override
	public void setEnvironment(final Environment env) {
		this.m_env = env;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String ignoreCaseForNamesTipText() {
		return "Ignore case when matching attribute names and nomina values.";
	}

	/**
	 * Set whether to ignore case when matching attribute names and nominal values.
	 *
	 * @param ignore
	 *            true if case is to be ignored
	 */
	public void setIgnoreCaseForNames(final boolean ignore) {
		this.m_ignoreCase = ignore;
	}

	/**
	 * Get whether to ignore case when matching attribute names and nominal values.
	 *
	 * @return true if case is to be ignored.
	 */
	public boolean getIgnoreCaseForNames() {
		return this.m_ignoreCase;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String trimTipText() {
		return "Trim white space from each end of attribute names and " + "nominal values before matching.";
	}

	/**
	 * Set whether to trim white space from each end of names before matching.
	 *
	 * @param trim
	 *            true to trim white space.
	 */
	public void setTrim(final boolean trim) {
		this.m_trim = trim;
	}

	/**
	 * Get whether to trim white space from each end of names before matching.
	 *
	 * @return true if white space is to be trimmed.
	 */
	public boolean getTrim() {
		return this.m_trim;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String suppressMappingReportTipText() {
		return "Don't output a report of model-to-input mappings.";
	}

	/**
	 * Set whether to suppress output the report of model to input mappings.
	 *
	 * @param suppress
	 *            true to suppress this output.
	 */
	public void setSuppressMappingReport(final boolean suppress) {
		this.m_suppressMappingReport = suppress;
	}

	/**
	 * Get whether to suppress output the report of model to input mappings.
	 *
	 * @return true if this output is to be suppressed.
	 */
	public boolean getSuppressMappingReport() {
		return this.m_suppressMappingReport;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String modelPathTipText() {
		return "Set the path from which to load a model. " + "Loading occurs when the first test instance " + "is received. Environment variables can be used in the " + "supplied path.";
	}

	/**
	 * Set the path from which to load a model. Loading occurs when the first test instance is received or getModelHeader() is called programatically. Environment variables can be used in the supplied path - e.g. ${HOME}/myModel.model.
	 *
	 * @param modelPath
	 *            the path to the model to load.
	 * @throws Exception
	 *             if a problem occurs during loading.
	 */
	public void setModelPath(final String modelPath) throws Exception {
		if (this.m_env == null) {
			this.m_env = Environment.getSystemWide();
		}

		this.m_modelPath = modelPath;

		// loadModel(modelPath);
	}

	/**
	 * Get the path used for loading a model.
	 *
	 * @return the path used for loading a model.
	 */
	public String getModelPath() {
		return this.m_modelPath;
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();

		result.disable(Capability.RELATIONAL_ATTRIBUTES);

		return result;
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 *
	 * <pre>
	 * -I
	 *  Ignore case when matching attribute names and nominal values.
	 * </pre>
	 *
	 * <pre>
	 * -M
	 *  Suppress the output of the mapping report.
	 * </pre>
	 *
	 * <pre>
	 * -trim
	 *  Trim white space from either end of names before matching.
	 * </pre>
	 *
	 * <pre>
	 * -L &lt;path to model to load&gt;
	 *  Path to a model to load. If set, this model
	 *  will be used for prediction and any base classifier
	 *  specification will be ignored. Environment variables
	 *  may be used in the path (e.g. ${HOME}/myModel.model)
	 * </pre>
	 *
	 * <pre>
	 * -D
	 *  If set, classifier is run in debug mode and
	 *  may output additional info to the console
	 * </pre>
	 *
	 * <pre>
	 * -W
	 *  Full name of base classifier.
	 *  (default: weka.classifiers.rules.ZeroR)
	 * </pre>
	 *
	 * <pre>
	 * Options specific to classifier weka.classifiers.rules.ZeroR:
	 * </pre>
	 *
	 * <pre>
	 * -D
	 *  If set, classifier is run in debug mode and
	 *  may output additional info to the console
	 * </pre>
	 *
	 * <!-- options-end -->
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(4);

		newVector.addElement(new Option("\tIgnore case when matching attribute " + "names and nominal values.", "I", 0, "-I"));
		newVector.addElement(new Option("\tSuppress the output of the mapping report.", "M", 0, "-M"));
		newVector.addElement(new Option("\tTrim white space from either end of names " + "before matching.", "trim", 0, "-trim"));
		newVector.addElement(new Option("\tPath to a model to load. If set, this model" + "\n\twill be used for prediction and any base classifier" + "\n\tspecification will be ignored. Environment variables"
				+ "\n\tmay be used in the path (e.g. ${HOME}/myModel.model)", "L", 1, "-L <path to model to load>"));

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
	 * -I
	 *  Ignore case when matching attribute names and nominal values.
	 * </pre>
	 *
	 * <pre>
	 * -M
	 *  Suppress the output of the mapping report.
	 * </pre>
	 *
	 * <pre>
	 * -trim
	 *  Trim white space from either end of names before matching.
	 * </pre>
	 *
	 * <pre>
	 * -L &lt;path to model to load&gt;
	 *  Path to a model to load. If set, this model
	 *  will be used for prediction and any base classifier
	 *  specification will be ignored. Environment variables
	 *  may be used in the path (e.g. ${HOME}/myModel.model)
	 * </pre>
	 *
	 * <pre>
	 * -D
	 *  If set, classifier is run in debug mode and
	 *  may output additional info to the console
	 * </pre>
	 *
	 * <pre>
	 * -W
	 *  Full name of base classifier.
	 *  (default: weka.classifiers.rules.ZeroR)
	 * </pre>
	 *
	 * <pre>
	 * Options specific to classifier weka.classifiers.rules.ZeroR:
	 * </pre>
	 *
	 * <pre>
	 * -D
	 *  If set, classifier is run in debug mode and
	 *  may output additional info to the console
	 * </pre>
	 *
	 * <!-- options-end -->
	 *
	 * Options after -- are passed to the designated classifier.
	 * <p>
	 *
	 * @param options
	 *            the list of options as an array of strings
	 * @throws Exception
	 *             if an option is not supported
	 */
	@Override
	public void setOptions(final String[] options) throws Exception {
		this.setIgnoreCaseForNames(Utils.getFlag('I', options));
		this.setSuppressMappingReport(Utils.getFlag('M', options));
		this.setTrim(Utils.getFlag("trim", options));

		String modelPath = Utils.getOption('L', options);
		if (modelPath.length() > 0) {
			this.setModelPath(modelPath);
		}

		super.setOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {
		String[] superOptions = super.getOptions();
		String[] options = new String[superOptions.length + 5];

		int current = 0;
		if (this.getIgnoreCaseForNames()) {
			options[current++] = "-I";
		}
		if (this.getSuppressMappingReport()) {
			options[current++] = "-M";
		}
		if (this.getTrim()) {
			options[current++] = "-trim";
		}

		if (this.getModelPath() != null && this.getModelPath().length() > 0) {
			options[current++] = "-L";
			options[current++] = this.getModelPath();
		}

		System.arraycopy(superOptions, 0, options, current, superOptions.length);

		current += superOptions.length;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	/**
	 * Set the test structure (if known in advance) that we are likely to see. If set, then a call to buildClassifier() will not overwrite any test structure that has been recorded with the current training structure. This is useful for
	 * getting a correct mapping report output in toString() after buildClassifier has been called and before any test instance has been seen. Test structure and mapping will get reset if a test instance is received whose structure does not
	 * match the recorded test structure.
	 *
	 * @param testStructure
	 *            the structure of the test instances that we are likely to see (if known in advance)
	 */
	public void setTestStructure(final Instances testStructure) {
		this.m_inputHeader = testStructure;
		this.m_initialTestStructureKnown = true;
	}

	/**
	 * Set the structure of the data used to create the model. This method is useful for clients who have an existing in-memory model that they'd like to wrap in the InputMappedClassifier
	 *
	 * @param modelHeader
	 *            the structure of the data used to build the wrapped model
	 */
	public void setModelHeader(final Instances modelHeader) {
		this.m_modelHeader = modelHeader;
	}

	private void loadModel(String modelPath) throws Exception {
		if (modelPath != null && modelPath.length() > 0) {
			try {
				if (this.m_env == null) {
					this.m_env = Environment.getSystemWide();
				}

				modelPath = this.m_env.substitute(modelPath);
			} catch (Exception ex) {
				// ignore any problems
			}

			try {
				Object[] modelAndHeader = SerializationHelper.readAll(modelPath);

				if (modelAndHeader.length != 2) {
					throw new Exception("[InputMappedClassifier] serialized model file " + "does not seem to contain both a model and " + "the instances header used in training it!");
				} else {
					this.setClassifier((Classifier) modelAndHeader[0]);
					this.m_modelHeader = (Instances) modelAndHeader[1];
				}
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}
	}

	/**
	 * Build the classifier
	 *
	 * @param data
	 *            the training data to be used for generating the bagged classifier.
	 * @throws Exception
	 *             if the classifier could not be built successfully
	 */
	@Override
	public void buildClassifier(final Instances data) throws Exception {
		if (!this.m_initialTestStructureKnown) {
			this.m_inputHeader = new Instances(data, 0);
		}

		this.m_attributeMap = null;

		if (this.m_modelPath != null && this.m_modelPath.length() > 0) {
			return; // Don't build a classifier if a path has been specified
		}

		// can classifier handle the data?
		this.getCapabilities().testWithFail(data);

		this.m_Classifier.buildClassifier(data);
		// m_loadedClassifier = m_Classifier;
		this.m_modelHeader = new Instances(data, 0);
	}

	private boolean stringMatch(String one, String two) {
		if (this.m_trim) {
			one = one.trim();
			two = two.trim();
		}

		if (this.m_ignoreCase) {
			return one.equalsIgnoreCase(two);
		} else {
			return one.equals(two);
		}
	}

	/**
	 * Helper method to pad/truncate strings
	 *
	 * @param s
	 *            String to modify
	 * @param pad
	 *            character to pad with
	 * @param len
	 *            length of final string
	 * @return final String
	 */
	private String getFixedLengthString(final String s, final char pad, final int len) {

		String padded = null;
		if (len <= 0) {
			return s;
		}
		// truncate?
		if (s.length() >= len) {
			return s.substring(0, len);
		} else {
			char[] buf = new char[len - s.length()];
			for (int j = 0; j < len - s.length(); j++) {
				buf[j] = pad;
			}
			padded = s + new String(buf);
		}

		return padded;
	}

	private StringBuffer createMappingReport() {
		StringBuffer result = new StringBuffer();
		result.append("Attribute mappings:\n\n");

		int maxLength = 0;
		for (int i = 0; i < this.m_modelHeader.numAttributes(); i++) {
			if (this.m_modelHeader.attribute(i).name().length() > maxLength) {
				maxLength = this.m_modelHeader.attribute(i).name().length();
			}
		}
		maxLength += 12;

		int minLength = 16;
		String headerS = "Model attributes";
		String sep = "----------------";

		if (maxLength < minLength) {
			maxLength = minLength;
		}

		headerS = this.getFixedLengthString(headerS, ' ', maxLength);
		sep = this.getFixedLengthString(sep, '-', maxLength);
		sep += "\t    ----------------\n";
		headerS += "\t    Incoming attributes\n";
		result.append(headerS);
		result.append(sep);

		for (int i = 0; i < this.m_modelHeader.numAttributes(); i++) {
			Attribute temp = this.m_modelHeader.attribute(i);
			String attName = "(" + ((temp.isNumeric()) ? "numeric)" : "nominal)") + " " + temp.name();
			attName = this.getFixedLengthString(attName, ' ', maxLength);
			attName += "\t--> ";
			result.append(attName);
			String inAttNum = "";
			if (this.m_attributeStatus[i] == NO_MATCH) {
				inAttNum += "- ";
				result.append(inAttNum + "missing (no match)\n");
			} else if (this.m_attributeStatus[i] == TYPE_MISMATCH) {
				inAttNum += (this.m_attributeMap[i] + 1) + " ";
				result.append(inAttNum + "missing (type mis-match)\n");
			} else {
				Attribute inAtt = this.m_inputHeader.attribute(this.m_attributeMap[i]);
				String inName = "" + (this.m_attributeMap[i] + 1) + " (" + ((inAtt.isNumeric()) ? "numeric)" : "nominal)") + " " + inAtt.name();
				result.append(inName + "\n");
			}
		}

		return result;
	}

	protected static final int NO_MATCH = -1;
	protected static final int TYPE_MISMATCH = -2;
	protected static final int OK = -3;

	private boolean regenerateMapping() throws Exception {
		this.loadModel(this.m_modelPath); // load a model (if specified)

		if (this.m_modelHeader == null) {
			return false;
		}

		this.m_attributeMap = new int[this.m_modelHeader.numAttributes()];
		this.m_attributeStatus = new int[this.m_modelHeader.numAttributes()];
		this.m_nominalValueMap = new int[this.m_modelHeader.numAttributes()][];

		for (int i = 0; i < this.m_modelHeader.numAttributes(); i++) {
			String modelAttName = this.m_modelHeader.attribute(i).name();
			this.m_attributeStatus[i] = NO_MATCH;

			for (int j = 0; j < this.m_inputHeader.numAttributes(); j++) {
				String incomingAttName = this.m_inputHeader.attribute(j).name();
				if (this.stringMatch(modelAttName, incomingAttName)) {
					this.m_attributeMap[i] = j;
					this.m_attributeStatus[i] = OK;

					Attribute modelAtt = this.m_modelHeader.attribute(i);
					Attribute incomingAtt = this.m_inputHeader.attribute(j);

					// check types
					if (modelAtt.type() != incomingAtt.type()) {
						this.m_attributeStatus[i] = TYPE_MISMATCH;
						break;
					}

					// now check nominal values (number, names...)
					if (modelAtt.numValues() != incomingAtt.numValues()) {
						System.out.println("[InputMappedClassifier] Warning: incoming nominal " + "attribute " + incomingAttName + " does not have the same " + "number of values as model attribute " + modelAttName);

					}

					if (modelAtt.isNominal() && incomingAtt.isNominal()) {
						int[] valuesMap = new int[incomingAtt.numValues()];
						for (int k = 0; k < incomingAtt.numValues(); k++) {
							String incomingNomValue = incomingAtt.value(k);
							int indexInModel = modelAtt.indexOfValue(incomingNomValue);
							if (indexInModel < 0) {
								valuesMap[k] = NO_MATCH;
							} else {
								valuesMap[k] = indexInModel;
							}
						}
						this.m_nominalValueMap[i] = valuesMap;
					}
				}
			}
		}

		return true;
	}

	/**
	 * Return the instance structure that the encapsulated model was built with. If the classifier will be built from scratch by InputMappedClassifier then this method just returns the default structure that is passed in as argument.
	 *
	 * @param defaultH
	 *            the default instances structure
	 * @return the instances structure used to create the encapsulated model
	 * @throws Exception
	 *             if a problem occurs
	 */
	public Instances getModelHeader(final Instances defaultH) throws Exception {
		this.loadModel(this.m_modelPath);

		// If the model header is null, then we must be going to build from
		// scratch in buildClassifier. Therefore, just return the supplied default,
		// since this has to match what we will build with
		Instances toReturn = (this.m_modelHeader == null) ? defaultH : this.m_modelHeader;

		return new Instances(toReturn, 0);
	}

	// get the mapped class index (i.e. the index in the incoming data of
	// the attribute that the model uses as the class
	public int getMappedClassIndex() throws Exception {
		if (this.m_modelHeader == null) {
			throw new Exception("[InputMappedClassifier] No model available!");
		}

		if (this.m_attributeMap[this.m_modelHeader.classIndex()] == NO_MATCH) {
			return -1;
		}

		return this.m_attributeMap[this.m_modelHeader.classIndex()];
	}

	public synchronized Instance constructMappedInstance(final Instance incoming) throws Exception {

		boolean regenerateMapping = false;

		if (this.m_inputHeader == null) {
			this.m_inputHeader = incoming.dataset();
			regenerateMapping = true;
			this.m_initialTestStructureKnown = false;
		} else if (!this.m_inputHeader.equalHeaders(incoming.dataset())) {
			this.m_inputHeader = incoming.dataset();

			regenerateMapping = true;
			this.m_initialTestStructureKnown = false;
		} else if (this.m_attributeMap == null) {
			regenerateMapping = true;
			this.m_initialTestStructureKnown = false;
		}

		if (regenerateMapping) {
			this.regenerateMapping();
			this.m_vals = null;

			if (!this.m_suppressMappingReport) {
				StringBuffer result = this.createMappingReport();
			}
		}

		this.m_vals = new double[this.m_modelHeader.numAttributes()];

		for (int i = 0; i < this.m_modelHeader.numAttributes(); i++) {
			if (this.m_attributeStatus[i] == OK) {
				Attribute modelAtt = this.m_modelHeader.attribute(i);
				this.m_inputHeader.attribute(this.m_attributeMap[i]);

				if (Utils.isMissingValue(incoming.value(this.m_attributeMap[i]))) {
					this.m_vals[i] = Utils.missingValue();
					continue;
				}

				if (modelAtt.isNumeric()) {
					this.m_vals[i] = incoming.value(this.m_attributeMap[i]);
				} else if (modelAtt.isNominal()) {
					int mapVal = this.m_nominalValueMap[i][(int) incoming.value(this.m_attributeMap[i])];

					if (mapVal == NO_MATCH) {
						this.m_vals[i] = Utils.missingValue();
					} else {
						this.m_vals[i] = mapVal;
					}
				}
			} else {
				this.m_vals[i] = Utils.missingValue();
			}
		}

		Instance newInst = new DenseInstance(incoming.weight(), this.m_vals);
		newInst.setDataset(this.m_modelHeader);

		return newInst;
	}

	@Override
	public double classifyInstance(final Instance inst) throws Exception {
		Instance converted = this.constructMappedInstance(inst);
		return this.m_Classifier.classifyInstance(converted);
	}

	@Override
	public double[] distributionForInstance(final Instance inst) throws Exception {

		Instance converted = this.constructMappedInstance(inst);
		return this.m_Classifier.distributionForInstance(converted);
	}

	@Override
	public String toString() {
		StringBuffer buff = new StringBuffer();

		buff.append("InputMappedClassifier:\n\n");

		try {
			this.loadModel(this.m_modelPath);
		} catch (Exception ex) {
			return "[InputMappedClassifier] Problem loading model.";
		}

		if (this.m_modelPath != null && this.m_modelPath.length() > 0) {
			buff.append("Model sourced from: " + this.m_modelPath + "\n\n");
		}

		/*
		 * if (m_loadedClassifier != null) { buff.append(m_loadedClassifier); } else
		 * {
		 */
		buff.append(this.m_Classifier);
		// }

		if (!this.m_suppressMappingReport && this.m_inputHeader != null) {
			try {
				this.regenerateMapping();
			} catch (Exception ex) {
				ex.printStackTrace();
				return "[InputMappedClassifier] Problem loading model.";
			}
			if (this.m_attributeMap != null) {
				buff.append("\n" + this.createMappingReport().toString());
			}
		}

		return buff.toString();
	}

	/**
	 * Returns the type of graph this classifier represents.
	 *
	 * @return the type of graph
	 */
	@Override
	public int graphType() {

		if (this.m_Classifier instanceof Drawable) {
			return ((Drawable) this.m_Classifier).graphType();
		} else {
			return Drawable.NOT_DRAWABLE;
		}
	}

	/**
	 * Returns an enumeration of the additional measure names
	 *
	 * @return an enumeration of the measure names
	 */
	@Override
	public Enumeration<String> enumerateMeasures() {
		Vector<String> newVector = new Vector<String>();

		if (this.m_Classifier instanceof AdditionalMeasureProducer) {
			Enumeration<String> en = ((AdditionalMeasureProducer) this.m_Classifier).enumerateMeasures();
			while (en.hasMoreElements()) {
				String mname = en.nextElement();
				newVector.addElement(mname);
			}
		}
		return newVector.elements();
	}

	/**
	 * Returns the value of the named measure
	 *
	 * @param additionalMeasureName
	 *            the name of the measure to query for its value
	 * @return the value of the named measure
	 * @throws IllegalArgumentException
	 *             if the named measure is not supported
	 */
	@Override
	public double getMeasure(final String additionalMeasureName) {
		if (this.m_Classifier instanceof AdditionalMeasureProducer) {
			return ((AdditionalMeasureProducer) this.m_Classifier).getMeasure(additionalMeasureName);
		} else {
			throw new IllegalArgumentException(additionalMeasureName + " not supported (InputMappedClassifier)");
		}
	}

	/**
	 * Returns graph describing the classifier (if possible).
	 *
	 * @return the graph of the classifier in dotty format
	 * @throws Exception
	 *             if the classifier cannot be graphed
	 */
	@Override
	public String graph() throws Exception {

		if (this.m_Classifier != null && this.m_Classifier instanceof Drawable) {
			return ((Drawable) this.m_Classifier).graph();
		} else {
			throw new Exception("Classifier: " + this.getClassifierSpec() + " cannot be graphed");
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

	/**
	 * Main method for testing this class.
	 *
	 * @param argv
	 *            should contain the following arguments: -t training file [-T test file] [-c class index]
	 */
	public static void main(final String[] argv) {
		runClassifier(new InputMappedClassifier(), argv);
	}

}
