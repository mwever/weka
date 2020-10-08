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
 *    Saver.java
 *    Copyright (C) 2015 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.knowledgeflow.steps;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import weka.core.EnvironmentHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializedObject;
import weka.core.Utils;
import weka.core.WekaException;
import weka.core.converters.DatabaseConverter;
import weka.core.converters.DatabaseSaver;
import weka.gui.ProgrammaticProperty;
import weka.gui.knowledgeflow.StepVisual;
import weka.knowledgeflow.Data;
import weka.knowledgeflow.StepManager;
import weka.knowledgeflow.StepManagerImpl;

/**
 * Step that wraps weka.core.converters.Saver classes
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision: $
 */
@KFStep(name = "Saver", category = "DataSinks", toolTipText = "Weka saver wrapper", iconPath = "")
public class Saver extends WekaAlgorithmWrapper implements Serializable {

	private static final long serialVersionUID = 6831606284211403465L;

	/**
	 * Holds the structure
	 */
	protected Instances m_structure;

	/** The actual saver instance to use */
	protected weka.core.converters.Saver m_saver;

	/** True if the saver is a DatabaseSaver */
	protected boolean m_isDBSaver;

	/**
	 * For file-based savers - if true (default), relation name is used as the
	 * primary part of the filename. If false, then the prefix is used as the
	 * filename. Useful for preventing filenames from getting too long when there
	 * are many filters in a flow.
	 */
	private boolean m_relationNameForFilename = true;

	/**
	 * Get the class of the wrapped algorithm
	 *
	 * @return the class of the wrapped algorithm
	 */
	@Override
	public Class getWrappedAlgorithmClass() {
		return weka.core.converters.Saver.class;
	}

	/**
	 * Set the actual wrapped algorithm instance
	 *
	 * @param algo the wrapped algorithm instance
	 */
	@Override
	public void setWrappedAlgorithm(final Object algo) {
		super.setWrappedAlgorithm(algo);
		this.m_defaultIconPath = StepVisual.BASE_ICON_PATH + "DefaultDataSink.gif";
	}

	/**
	 * Get the saver instance that is wrapped by this step. Convenience method
	 * that delegates to {@code getWrappedAlgorithm()}
	 *
	 * @return the saver instance that is wrapped by this step
	 */
	public weka.core.converters.Saver getSaver() {
		return (weka.core.converters.Saver) this.getWrappedAlgorithm();
	}

	/**
	 * Set the saver instance that is wrapped by this step. Convenience method
	 * that delegates to {@code setWrappedAlgorithm()}.
	 *
	 * @param saver the saver instance that is wrapped by this step
	 */
	@ProgrammaticProperty
	public void setSaver(final weka.core.converters.Saver saver) {
		this.setWrappedAlgorithm(saver);
	}

	/**
	 * Get whether the relation name is the primary part of the filename.
	 *
	 * @return true if the relation name is part of the filename.
	 */
	public boolean getRelationNameForFilename() {
		return this.m_relationNameForFilename;
	}

	/**
	 * Set whether to use the relation name as the primary part of the filename.
	 * If false, then the prefix becomes the filename.
	 *
	 * @param r true if the relation name is to be part of the filename.
	 */
	public void setRelationNameForFilename(final boolean r) {
		this.m_relationNameForFilename = r;
	}

	/**
	 * Initialize the step
	 *
	 * @throws WekaException if a problem occurs during initialization
	 */
	@Override
	public void stepInit() throws WekaException {
		this.m_saver = null;

		if (!(this.getWrappedAlgorithm() instanceof weka.core.converters.Saver)) {
			throw new WekaException("Incorrect type of algorithm");
		}

		if (this.getWrappedAlgorithm() instanceof DatabaseConverter) {
			this.m_isDBSaver = true;
		}

		int numNonInstanceInputs = this.getStepManager().numIncomingConnectionsOfType(StepManager.CON_DATASET) + this.getStepManager().numIncomingConnectionsOfType(StepManager.CON_TRAININGSET)
				+ this.getStepManager().numIncomingConnectionsOfType(StepManager.CON_TESTSET);

		int numInstanceInput = this.getStepManager().numIncomingConnectionsOfType(StepManager.CON_INSTANCE);

		if (numNonInstanceInputs > 0 && numInstanceInput > 0) {
			WekaException cause = new WekaException("Can't have both instance and batch-based incomming connections!");
			cause.fillInStackTrace();
			this.getStepManager().logError(cause.getMessage(), cause);
			throw new WekaException(cause);
		}
	}

	/**
	 * Save a batch of instances
	 *
	 * @param data the {@code Instances} to save
	 * @param setNum the set/fold number of this batch
	 * @param maxSetNum the maximum number of sets/folds in this batch
	 * @param connectionName the connection type that this batch arrived in
	 * @throws WekaException if a problem occurs
	 */
	protected void saveBatch(final Instances data, final Integer setNum, final Integer maxSetNum, final String connectionName) throws WekaException {
		this.getStepManager().processing();

		try {
			weka.core.converters.Saver saver = (weka.core.converters.Saver) new SerializedObject(this.m_saver).getObject();
			if (this.m_saver instanceof EnvironmentHandler) {
				((EnvironmentHandler) saver).setEnvironment(this.getStepManager().getExecutionEnvironment().getEnvironmentVariables());
			}

			String fileName = this.sanitizeFilename(data.relationName());

			String additional = setNum != null && (setNum + maxSetNum != 2) ? "_" + connectionName + "_" + setNum + "_of_" + maxSetNum : "";

			if (!this.m_isDBSaver) {
				saver.setDirAndPrefix(fileName, additional);
			} else {
				if (((DatabaseSaver) saver).getRelationForTableName()) {
					((DatabaseSaver) saver).setTableName(fileName);
				}
				((DatabaseSaver) saver).setRelationForTableName(false);
				String setName = ((DatabaseSaver) saver).getTableName();
				setName = setName.replaceFirst("_" + connectionName + "_[0-9]+_of_[0-9]+", "");
				((DatabaseSaver) saver).setTableName(setName + additional);
			}
			saver.setInstances(data);

			this.getStepManager().logBasic("Saving " + data.relationName() + additional);
			this.getStepManager().statusMessage("Saving " + data.relationName() + additional);
			saver.writeBatch();

			if (!this.isStopRequested()) {
				this.getStepManager().logBasic("Save successful");
				this.getStepManager().statusMessage("Finished.");
			} else {
				this.getStepManager().interrupted();
			}
		} catch (Exception ex) {
			WekaException e = new WekaException(ex);
			// e.printStackTrace();
			throw e;
		} finally {
			this.getStepManager().finished();
		}
	}

	/**
	 * Processes incoming data
	 *
	 * @param data the data process
	 * @throws WekaException if a problem occurs
	 */
	@Override
	public synchronized void processIncoming(final Data data) throws WekaException {

		if (this.m_saver == null) {
			try {
				this.m_saver = (weka.core.converters.Saver) new SerializedObject(this.getWrappedAlgorithm()).getObject();
			} catch (Exception ex) {
				throw new WekaException(ex);
			}

			if (this.m_saver instanceof EnvironmentHandler) {
				((EnvironmentHandler) this.m_saver).setEnvironment(this.getStepManager().getExecutionEnvironment().getEnvironmentVariables());
			}

			if (data.getConnectionName().equalsIgnoreCase(StepManager.CON_INSTANCE)) {
				// incremental saving
				Instance forStructure = (Instance) data.getPayloadElement(StepManager.CON_INSTANCE);
				if (forStructure != null) {
					// processing();
					this.m_saver.setRetrieval(weka.core.converters.Saver.INCREMENTAL);
					String fileName = this.sanitizeFilename(forStructure.dataset().relationName());
					try {
						this.m_saver.setDirAndPrefix(fileName, "");
					} catch (Exception ex) {
						throw new WekaException(ex);
					}
					try {
						this.m_saver.setInstances(forStructure.dataset());
					} catch (InterruptedException e) {
						e.printStackTrace();
					}

					if (this.m_isDBSaver) {
						if (((DatabaseSaver) this.m_saver).getRelationForTableName()) {
							((DatabaseSaver) this.m_saver).setTableName(fileName);
							((DatabaseSaver) this.m_saver).setRelationForTableName(false);
						}
					}
				}
			}
		}

		if (data.getConnectionName().equals(StepManager.CON_DATASET) || data.getConnectionName().equals(StepManager.CON_TRAININGSET) || data.getConnectionName().equals(StepManager.CON_TESTSET)) {
			this.m_saver.setRetrieval(weka.core.converters.Saver.BATCH);
			Instances theData = (Instances) data.getPayloadElement(data.getConnectionName());
			Integer setNum = (Integer) data.getPayloadElement(StepManager.CON_AUX_DATA_SET_NUM);
			Integer maxSetNum = (Integer) data.getPayloadElement(StepManager.CON_AUX_DATA_MAX_SET_NUM);

			this.saveBatch(theData, setNum, maxSetNum, data.getConnectionName());

			return;
		}

		Instance toSave = (Instance) data.getPayloadElement(StepManager.CON_INSTANCE);
		boolean streamEnd = this.getStepManager().isStreamFinished(data);
		try {
			if (streamEnd) {
				this.m_saver.writeIncremental(null);
				this.getStepManager().throughputFinished(new Data(StepManagerImpl.CON_INSTANCE));
				return;
			}

			if (!this.isStopRequested()) {
				this.getStepManager().throughputUpdateStart();
				this.m_saver.writeIncremental(toSave);
			} else {
				// make sure that saver finishes and closes file
				this.m_saver.writeIncremental(null);
			}
			this.getStepManager().throughputUpdateEnd();
		} catch (Exception ex) {
			throw new WekaException(ex);
		}
	}

	/**
	 * Get a list of incoming connection types that this step can receive at this
	 * time
	 *
	 * @return a list of incoming connection types
	 */
	@Override
	public List<String> getIncomingConnectionTypes() {

		int numInstance = this.getStepManager().getIncomingConnectedStepsOfConnectionType(StepManager.CON_INSTANCE).size();

		int numNonInstance = this.getStepManager().getIncomingConnectedStepsOfConnectionType(StepManager.CON_DATASET).size() + this.getStepManager().getIncomingConnectedStepsOfConnectionType(StepManager.CON_TRAININGSET).size()
				+ this.getStepManager().getIncomingConnectedStepsOfConnectionType(StepManager.CON_TESTSET).size();

		if (numInstance + numNonInstance == 0) {
			return Arrays.asList(StepManager.CON_DATASET, StepManager.CON_TRAININGSET, StepManager.CON_TESTSET, StepManager.CON_INSTANCE);
		}

		return new ArrayList<String>();
	}

	/**
	 * Get a list of outgoing connection types that this step can produce at this
	 * time
	 *
	 * @return a list of outgoing connection types
	 */
	@Override
	public List<String> getOutgoingConnectionTypes() {
		// no outgoing connections
		return new ArrayList<String>();
	}

	/**
	 * makes sure that the filename is valid, i.e., replaces slashes, backslashes
	 * and colons with underscores ("_"). Also try to prevent filename from
	 * becoming insanely long by removing package part of class names.
	 * 
	 * @param filename the filename to cleanse
	 * @return the cleansed filename
	 */
	protected String sanitizeFilename(String filename) {
		filename = filename.replaceAll("\\\\", "_").replaceAll(":", "_").replaceAll("/", "_");
		filename = Utils.removeSubstring(filename, "weka.filters.supervised.instance.");
		filename = Utils.removeSubstring(filename, "weka.filters.supervised.attribute.");
		filename = Utils.removeSubstring(filename, "weka.filters.unsupervised.instance.");
		filename = Utils.removeSubstring(filename, "weka.filters.unsupervised.attribute.");
		filename = Utils.removeSubstring(filename, "weka.clusterers.");
		filename = Utils.removeSubstring(filename, "weka.associations.");
		filename = Utils.removeSubstring(filename, "weka.attributeSelection.");
		filename = Utils.removeSubstring(filename, "weka.estimators.");
		filename = Utils.removeSubstring(filename, "weka.datagenerators.");

		if (!this.m_isDBSaver && !this.m_relationNameForFilename) {
			filename = "";
			try {
				if (this.m_saver.filePrefix().equals("")) {
					this.m_saver.setFilePrefix("no-name");
				}
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}

		return filename;
	}

	/**
	 * Return the fully qualified name of a custom editor component (JComponent)
	 * to use for editing the properties of the step. This method can return null,
	 * in which case the system will dynamically generate an editor using the
	 * GenericObjectEditor
	 *
	 * @return the fully qualified name of a step editor component
	 */
	@Override
	public String getCustomEditorForStep() {
		return "weka.gui.knowledgeflow.steps.SaverStepEditorDialog";
	}
}
