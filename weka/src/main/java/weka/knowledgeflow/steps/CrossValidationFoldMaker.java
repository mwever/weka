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
 *    CrossValidationFoldMaker.java
 *    Copyright (C) 2015 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.knowledgeflow.steps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import weka.core.Instances;
import weka.core.OptionMetadata;
import weka.core.WekaException;
import weka.gui.knowledgeflow.KFGUIConsts;
import weka.knowledgeflow.Data;
import weka.knowledgeflow.StepManager;

/**
 * Step for generating cross-validation splits
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision: $
 */
@KFStep(name = "CrossValidationFoldMaker", category = "Evaluation", toolTipText = "A Step that creates stratified cross-validation folds from incoming data", iconPath = KFGUIConsts.BASE_ICON_PATH + "CrossValidationFoldMaker.gif")
public class CrossValidationFoldMaker extends BaseStep {

	private static final long serialVersionUID = 6090713408437825355L;

	/** True to preserve order of the instances rather than randomly shuffling */
	protected boolean m_preserveOrder;

	/** User-specified number of folds */
	protected String m_numFoldsS = "10";

	/** User-specified random seed */
	protected String m_seedS = "1";

	/** Resolved number of folds */
	protected int m_numFolds = 10;

	/** Resolved random seed */
	protected long m_seed = 1L;

	/**
	 * Set the number of folds to create
	 *
	 * @param folds the number of folds to create
	 */
	@OptionMetadata(displayName = "Number of folds", description = "THe number of folds to create", displayOrder = 0)
	public void setNumFolds(final String folds) {
		this.m_numFoldsS = folds;
	}

	/**
	 * Get the number of folds to create
	 *
	 * @return the number of folds to create
	 */
	public String getNumFolds() {
		return this.m_numFoldsS;
	}

	/**
	 * Set whether to preserve the order of the input instances when creatinbg
	 * the folds
	 *
	 * @param preserve true to preserve the order
	 */
	@OptionMetadata(displayName = "Preserve instances order", description = "Preserve the order of instances rather than randomly shuffling", displayOrder = 1)
	public void setPreserveOrder(final boolean preserve) {
		this.m_preserveOrder = preserve;
	}

	/**
	 * Get whether to preserve the order of the input instances when creatinbg
	 * the folds
	 *
	 * @return true to preserve the order
	 */
	public boolean getPreserveOrder() {
		return this.m_preserveOrder;
	}

	/**
	 * Set the random seed to use
	 *
	 * @param seed the random seed to use
	 */
	@OptionMetadata(displayName = "Random seed", description = "The random seed to use for shuffling", displayOrder = 3)
	public void setSeed(final String seed) {
		this.m_seedS = seed;
	}

	/**
	 * Get the random seed
	 *
	 * @return the random seed
	 */
	public String getSeed() {
		return this.m_seedS;
	}

	/**
	 * Initialize the step.
	 *
	 * @throws WekaException if a problem occurs during initialization
	 */
	@Override
	public void stepInit() throws WekaException {
		String seed = this.getStepManager().environmentSubstitute(this.getSeed());
		try {
			this.m_seed = Long.parseLong(seed);
		} catch (NumberFormatException ex) {
			this.getStepManager().logWarning("Unable to parse seed value: " + seed);
		}

		String folds = this.getStepManager().environmentSubstitute(this.getNumFolds());
		try {
			this.m_numFolds = Integer.parseInt(folds);
		} catch (NumberFormatException e) {
			this.getStepManager().logWarning("Unable to parse number of folds value: " + folds);
		}
	}

	/**
	 * Process an incoming data payload (if the step accepts incoming connections)
	 *
	 * @param data the payload to process
	 * @throws WekaException if a problem occurs
	* @throws InterruptedException
	 */
	@Override
	public void processIncoming(final Data data) throws WekaException, InterruptedException {
		this.getStepManager().processing();
		String incomingConnName = data.getConnectionName();
		Instances dataSet = (Instances) data.getPayloadElement(incomingConnName);
		if (dataSet == null) {
			throw new WekaException("Incoming instances should not be null!");
		}
		dataSet = new Instances(dataSet);
		this.getStepManager().logBasic("Creating cross-validation folds");
		this.getStepManager().statusMessage("Creating cross-validation folds");

		Random random = new Random(this.m_seed);
		if (!this.getPreserveOrder()) {
			dataSet.randomize(random);
		}
		if (dataSet.classIndex() >= 0 && dataSet.attribute(dataSet.classIndex()).isNominal() && !this.getPreserveOrder()) {
			this.getStepManager().logBasic("Stratifying data");
			dataSet.stratify(this.m_numFolds);
		}

		for (int i = 0; i < this.m_numFolds; i++) {
			if (this.isStopRequested()) {
				break;
			}
			try {
				Instances train = (!this.m_preserveOrder) ? dataSet.trainCV(this.m_numFolds, i, random) : dataSet.trainCV(this.m_numFolds, i);
				Instances test = dataSet.testCV(this.m_numFolds, i);

				Data trainData = new Data(StepManager.CON_TRAININGSET);
				trainData.setPayloadElement(StepManager.CON_TRAININGSET, train);
				trainData.setPayloadElement(StepManager.CON_AUX_DATA_SET_NUM, i + 1);
				trainData.setPayloadElement(StepManager.CON_AUX_DATA_MAX_SET_NUM, this.m_numFolds);
				Data testData = new Data(StepManager.CON_TESTSET);
				testData.setPayloadElement(StepManager.CON_TESTSET, test);
				testData.setPayloadElement(StepManager.CON_AUX_DATA_SET_NUM, i + 1);
				testData.setPayloadElement(StepManager.CON_AUX_DATA_MAX_SET_NUM, this.m_numFolds);
				if (!this.isStopRequested()) {
					this.getStepManager().outputData(trainData, testData);
				}
			} catch (InterruptedException e) {
				throw new WekaException(e);
			}
		}

		this.getStepManager().finished();
	}

	/**
	 * Get a list of incoming connection types that this step can accept. Ideally
	 * (and if appropriate), this should take into account the state of the step
	 * and any existing incoming connections. E.g. a step might be able to accept
	 * one (and only one) incoming batch data connection.
	 *
	 * @return a list of incoming connections that this step can accept given its
	 *         current state
	 */
	@Override
	public List<String> getIncomingConnectionTypes() {
		if (this.getStepManager().numIncomingConnections() > 0) {
			return new ArrayList<String>();
		}

		return Arrays.asList(StepManager.CON_DATASET, StepManager.CON_TRAININGSET, StepManager.CON_TESTSET);
	}

	/**
	 * Get a list of outgoing connection types that this step can produce. Ideally
	 * (and if appropriate), this should take into account the state of the step
	 * and the incoming connections. E.g. depending on what incoming connection is
	 * present, a step might be able to produce a trainingSet output, a testSet
	 * output or neither, but not both.
	 *
	 * @return a list of outgoing connections that this step can produce
	 */
	@Override
	public List<String> getOutgoingConnectionTypes() {
		return this.getStepManager().numIncomingConnections() > 0 ? Arrays.asList(StepManager.CON_TRAININGSET, StepManager.CON_TESTSET) : new ArrayList<String>();
	}

	/**
	 * If possible, get the output structure for the named connection type as a
	 * header-only set of instances. Can return null if the specified connection
	 * type is not representable as Instances or cannot be determined at present.
	 *
	 * @param connectionName the name of the connection type to get the output
	 *          structure for
	 * @return the output structure as a header-only Instances object
	 * @throws WekaException if a problem occurs
	 */
	@Override
	public Instances outputStructureForConnectionType(final String connectionName) throws WekaException {

		// we produce training and testset connections
		if ((!connectionName.equals(StepManager.CON_TRAININGSET) && !connectionName.equals(StepManager.CON_TESTSET)) || this.getStepManager().numIncomingConnections() == 0) {
			return null;
		}

		// our output structure is the same as whatever kind of input we are getting
		Instances strucForDatasetCon = this.getStepManager().getIncomingStructureForConnectionType(StepManager.CON_DATASET);
		if (strucForDatasetCon != null) {
			return strucForDatasetCon;
		}

		Instances strucForTestsetCon = this.getStepManager().getIncomingStructureForConnectionType(StepManager.CON_TESTSET);
		if (strucForTestsetCon != null) {
			return strucForTestsetCon;
		}

		Instances strucForTrainingCon = this.getStepManager().getIncomingStructureForConnectionType(StepManager.CON_TRAININGSET);
		if (strucForTrainingCon != null) {
			return strucForTrainingCon;
		}

		return null;
	}
}
