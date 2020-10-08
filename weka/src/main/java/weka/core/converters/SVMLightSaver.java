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
 * SVMLightSaver.java
 * Copyright (C) 2006-2012 University of Waikato, Hamilton, NZ
 *
 */

package weka.core.converters;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.SingleIndex;
import weka.core.Utils;

/**
 * <!-- globalinfo-start --> Writes to a destination that is in svm light
 * format.<br/>
 * <br/>
 * For more information about svm light see:<br/>
 * <br/>
 * http://svmlight.joachims.org/
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -i &lt;the input file&gt;
 *  The input file
 * </pre>
 *
 * <pre>
 * -o &lt;the output file&gt;
 *  The output file
 * </pre>
 *
 * <pre>
 * -c &lt;class index&gt;
 *  The class index
 *  (default: last)
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 * @see Saver
 */
public class SVMLightSaver extends AbstractFileSaver implements BatchConverter, IncrementalConverter {

	/** for serialization. */
	private static final long serialVersionUID = 2605714599263995835L;

	/** the file extension. */
	public static String FILE_EXTENSION = SVMLightLoader.FILE_EXTENSION;

	/** the number of digits after the decimal point. */
	public static int MAX_DIGITS = 18;

	/** the class index. */
	protected SingleIndex m_ClassIndex = new SingleIndex("last");

	/**
	 * Constructor.
	 */
	public SVMLightSaver() {
		this.resetOptions();
	}

	/**
	 * Returns a string describing this Saver.
	 * 
	 * @return a description of the Saver suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Writes to a destination that is in svm light format.\n\n" + "For more information about svm light see:\n\n" + "http://svmlight.joachims.org/";
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();

		result.addElement(new Option("\tThe class index\n" + "\t(default: last)", "c", 1, "-c <class index>"));

		result.addAll(Collections.list(super.listOptions()));

		return result.elements();
	}

	/**
	 * returns the options of the current setup.
	 * 
	 * @return the current options
	 */
	@Override
	public String[] getOptions() {

		Vector<String> result = new Vector<String>();

		result.add("-c");
		result.add(this.getClassIndex());

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Parses the options for this object.
	 * <p/>
	 * 
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 * 
	 * <pre>
	 * -i &lt;the input file&gt;
	 *  The input file
	 * </pre>
	 * 
	 * <pre>
	 * -o &lt;the output file&gt;
	 *  The output file
	 * </pre>
	 * 
	 * <pre>
	 * -c &lt;class index&gt;
	 *  The class index
	 *  (default: last)
	 * </pre>
	 * 
	 * <!-- options-end -->
	 * 
	 * @param options the options to use
	 * @throws Exception if setting of options fails
	 */
	@Override
	public void setOptions(final String[] options) throws Exception {
		String tmpStr;

		tmpStr = Utils.getOption('c', options);
		if (tmpStr.length() != 0) {
			this.setClassIndex(tmpStr);
		} else {
			this.setClassIndex("last");
		}

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Returns a description of the file type.
	 * 
	 * @return a short file description
	 */
	@Override
	public String getFileDescription() {
		return "svm light data files";
	}

	/**
	 * Resets the Saver.
	 */
	@Override
	public void resetOptions() {
		super.resetOptions();
		this.setFileExtension(SVMLightLoader.FILE_EXTENSION);
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String classIndexTipText() {
		return "Sets the class index (\"first\" and \"last\" are valid values)";
	}

	/**
	 * Get the index of the class attribute.
	 * 
	 * @return the index of the class attribute
	 */
	public String getClassIndex() {
		return this.m_ClassIndex.getSingleIndex();
	}

	/**
	 * Sets index of the class attribute.
	 * 
	 * @param value the index of the class attribute
	 */
	public void setClassIndex(final String value) {
		this.m_ClassIndex.setSingleIndex(value);
	}

	/**
	 * Returns the Capabilities of this saver.
	 * 
	 * @return the capabilities of this object
	 * @see Capabilities
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);

		// class
		result.enable(Capability.BINARY_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.DATE_CLASS);

		return result;
	}

	/**
	 * Sets instances that should be stored.
	 * 
	 * @param instances the instances
	* @throws InterruptedException 
	 */
	@Override
	public void setInstances(final Instances instances) throws InterruptedException {
		this.m_ClassIndex.setUpper(instances.numAttributes() - 1);
		instances.setClassIndex(this.m_ClassIndex.getIndex());

		super.setInstances(instances);
	}

	/**
	 * turns the instance into a svm light row.
	 * 
	 * @param inst the instance to transform
	 * @return the generated svm light row
	 */
	protected String instanceToSvmlight(final Instance inst) {
		StringBuffer result;
		int i;

		result = new StringBuffer();

		// class
		if (inst.classAttribute().isNominal()) {
			if (inst.classValue() == 0) {
				result.append("1");
			} else if (inst.classValue() == 1) {
				result.append("-1");
			}
		} else {
			result.append("" + Utils.doubleToString(inst.classValue(), MAX_DIGITS));
		}

		// attributes
		for (i = 0; i < inst.numAttributes(); i++) {
			if (i == inst.classIndex()) {
				continue;
			}
			if (inst.value(i) == 0) {
				continue;
			}
			result.append(" " + (i + 1) + ":" + Utils.doubleToString(inst.value(i), MAX_DIGITS));
		}

		return result.toString();
	}

	/**
	 * Saves an instances incrementally. Structure has to be set by using the
	 * setStructure() method or setInstances() method.
	 * 
	 * @param inst the instance to save
	 * @throws IOException throws IOEXception if an instance cannot be saved
	 *           incrementally.
	 */
	@Override
	public void writeIncremental(final Instance inst) throws IOException {
		int writeMode = this.getWriteMode();
		Instances structure = this.getInstances();
		PrintWriter outW = null;

		if ((this.getRetrieval() == BATCH) || (this.getRetrieval() == NONE)) {
			throw new IOException("Batch and incremental saving cannot be mixed.");
		}

		if (this.getWriter() != null) {
			outW = new PrintWriter(this.getWriter());
		}

		if (writeMode == WAIT) {
			if (structure == null) {
				this.setWriteMode(CANCEL);
				if (inst != null) {
					System.err.println("Structure (Header Information) has to be set in advance");
				}
			} else {
				this.setWriteMode(STRUCTURE_READY);
			}
			writeMode = this.getWriteMode();
		}

		if (writeMode == CANCEL) {
			if (outW != null) {
				outW.close();
			}
			this.cancel();
		}

		// header
		if (writeMode == STRUCTURE_READY) {
			this.setWriteMode(WRITE);
			// no header
			writeMode = this.getWriteMode();
		}

		// row
		if (writeMode == WRITE) {
			if (structure == null) {
				throw new IOException("No instances information available.");
			}

			if (inst != null) {
				// write instance
				if ((this.retrieveFile() == null) && (outW == null)) {
					System.out.println(this.instanceToSvmlight(inst));
				} else {
					outW.println(this.instanceToSvmlight(inst));
					this.m_incrementalCounter++;
					// flush every 100 instances
					if (this.m_incrementalCounter > 100) {
						this.m_incrementalCounter = 0;
						outW.flush();
					}
				}
			} else {
				// close
				if (outW != null) {
					outW.flush();
					outW.close();
				}
				this.m_incrementalCounter = 0;
				this.resetStructure();
				outW = null;
				this.resetWriter();
			}
		}
	}

	/**
	 * Writes a Batch of instances.
	 * 
	 * @throws IOException throws IOException if saving in batch mode is not
	 *           possible
	 */
	@Override
	public void writeBatch() throws IOException {
		if (this.getInstances() == null) {
			throw new IOException("No instances to save");
		}

		if (this.getRetrieval() == INCREMENTAL) {
			throw new IOException("Batch and incremental saving cannot be mixed.");
		}

		this.setRetrieval(BATCH);
		this.setWriteMode(WRITE);

		if ((this.retrieveFile() == null) && (this.getWriter() == null)) {
			for (int i = 0; i < this.getInstances().numInstances(); i++) {
				System.out.println(this.instanceToSvmlight(this.getInstances().instance(i)));
			}
			this.setWriteMode(WAIT);
		} else {
			PrintWriter outW = new PrintWriter(this.getWriter());
			for (int i = 0; i < this.getInstances().numInstances(); i++) {
				outW.println(this.instanceToSvmlight(this.getInstances().instance(i)));
			}
			outW.flush();
			outW.close();
			this.setWriteMode(WAIT);
			outW = null;
			this.resetWriter();
			this.setWriteMode(CANCEL);
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
	 * Main method.
	 * 
	 * @param args should contain the options of a Saver.
	 */
	public static void main(final String[] args) {
		runFileSaver(new SVMLightSaver(), args);
	}
}
