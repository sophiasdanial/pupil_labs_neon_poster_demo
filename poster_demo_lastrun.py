#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Thu 24 Jul 09:41:25 2025
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard
from psychopy_eyetracker_pupil_labs.pupil_labs.stimuli import AprilTagFrameStim
from psychopy.tools.monitorunittools import convertToPix

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'poster_demo'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [2560, 1440]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/sophiastudent/Documents/GitHub/pupil_labs_neon_poster_demo/poster_demo_lastrun.py',
        savePickle=True, saveWideText=False,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Start" ---
    Ready = visual.TextStim(win=win, name='Ready',
        text='Get ready...',
        font='Arial',
        pos=(0, 0), draggable=True, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    distractor = visual.ShapeStim(
        win=win, name='distractor', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    tag_Frame_start = AprilTagFrameStim(
        win=win,
        name='tag_Frame_start', units='norm',
        contrast=1.0,
        h_count=4, v_count=3,
        marker_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], anchor='center',
        marker_size=0.125, marker_units='height',
        pos=(0, 0), size=[2, 2])
    
    # --- Initialize components for Routine "Poster" ---
    poster_scam = visual.ImageStim(
        win=win,
        name='poster_scam', 
        image='goisadvisorycsa-800.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.7, 0.7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    tagFrame = AprilTagFrameStim(
        win=win,
        name='tagFrame', units='norm',
        contrast=1.0,
        h_count=4, v_count=3,
        marker_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], anchor='center',
        marker_size=0.125, marker_units='height',
        pos=(0, 0), size=[2, 2])
    
    # --- Initialize components for Routine "Finish" ---
    End = visual.TextStim(win=win, name='End',
        text='Thank you for participating',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    distractor_end = visual.ShapeStim(
        win=win, name='distractor_end', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Start" ---
    # create an object to store info about Routine Start
    Start = data.Routine(
        name='Start',
        components=[Ready, distractor, tag_Frame_start],
    )
    Start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    if eyetracker is not None and hasattr(eyetracker, 'register_surface'):
        win_size_pix = convertToPix(np.array([2, 2]), [0, 0], 'norm', win)
        eyetracker.register_surface(tag_Frame_start.marker_verts, win_size_pix)
    # store start times for Start
    Start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Start.tStart = globalClock.getTime(format='float')
    Start.status = STARTED
    thisExp.addData('Start.started', Start.tStart)
    Start.maxDuration = 5.5
    # keep track of which components have finished
    StartComponents = Start.components
    for thisComponent in Start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start" ---
    Start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > Start.maxDuration-frameTolerance:
            Start.maxDurationReached = True
            continueRoutine = False
        
        # *Ready* updates
        
        # if Ready is starting this frame...
        if Ready.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Ready.frameNStart = frameN  # exact frame index
            Ready.tStart = t  # local t and not account for scr refresh
            Ready.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Ready, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Ready.started')
            # update status
            Ready.status = STARTED
            Ready.setAutoDraw(True)
        
        # if Ready is active this frame...
        if Ready.status == STARTED:
            # update params
            pass
        
        # if Ready is stopping this frame...
        if Ready.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Ready.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                Ready.tStop = t  # not accounting for scr refresh
                Ready.tStopRefresh = tThisFlipGlobal  # on global time
                Ready.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Ready.stopped')
                # update status
                Ready.status = FINISHED
                Ready.setAutoDraw(False)
        
        # *distractor* updates
        
        # if distractor is starting this frame...
        if distractor.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            distractor.frameNStart = frameN  # exact frame index
            distractor.tStart = t  # local t and not account for scr refresh
            distractor.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(distractor, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'distractor.started')
            # update status
            distractor.status = STARTED
            distractor.setAutoDraw(True)
        
        # if distractor is active this frame...
        if distractor.status == STARTED:
            # update params
            pass
        
        # if distractor is stopping this frame...
        if distractor.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > distractor.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                distractor.tStop = t  # not accounting for scr refresh
                distractor.tStopRefresh = tThisFlipGlobal  # on global time
                distractor.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'distractor.stopped')
                # update status
                distractor.status = FINISHED
                distractor.setAutoDraw(False)
        
        # *tag_Frame_start* updates
        
        # if tag_Frame_start is starting this frame...
        if tag_Frame_start.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            tag_Frame_start.frameNStart = frameN  # exact frame index
            tag_Frame_start.tStart = t  # local t and not account for scr refresh
            tag_Frame_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tag_Frame_start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'tag_Frame_start.started')
            # update status
            tag_Frame_start.status = STARTED
            tag_Frame_start.setAutoDraw(True)
        
        # if tag_Frame_start is active this frame...
        if tag_Frame_start.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Start.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start" ---
    for thisComponent in Start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Start
    Start.tStop = globalClock.getTime(format='float')
    Start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Start.stopped', Start.tStop)
    thisExp.nextEntry()
    # the Routine "Start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Poster" ---
    # create an object to store info about Routine Poster
    Poster = data.Routine(
        name='Poster',
        components=[poster_scam, tagFrame],
    )
    Poster.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    if eyetracker is not None and hasattr(eyetracker, 'register_surface'):
        win_size_pix = convertToPix(np.array([2, 2]), [0, 0], 'norm', win)
        eyetracker.register_surface(tagFrame.marker_verts, win_size_pix)
    # store start times for Poster
    Poster.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Poster.tStart = globalClock.getTime(format='float')
    Poster.status = STARTED
    thisExp.addData('Poster.started', Poster.tStart)
    Poster.maxDuration = 30
    # keep track of which components have finished
    PosterComponents = Poster.components
    for thisComponent in Poster.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Poster" ---
    Poster.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > Poster.maxDuration-frameTolerance:
            Poster.maxDurationReached = True
            continueRoutine = False
        
        # *poster_scam* updates
        
        # if poster_scam is starting this frame...
        if poster_scam.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            poster_scam.frameNStart = frameN  # exact frame index
            poster_scam.tStart = t  # local t and not account for scr refresh
            poster_scam.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(poster_scam, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'poster_scam.started')
            # update status
            poster_scam.status = STARTED
            poster_scam.setAutoDraw(True)
        
        # if poster_scam is active this frame...
        if poster_scam.status == STARTED:
            # update params
            pass
        
        # if poster_scam is stopping this frame...
        if poster_scam.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > poster_scam.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                poster_scam.tStop = t  # not accounting for scr refresh
                poster_scam.tStopRefresh = tThisFlipGlobal  # on global time
                poster_scam.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'poster_scam.stopped')
                # update status
                poster_scam.status = FINISHED
                poster_scam.setAutoDraw(False)
        
        # *tagFrame* updates
        
        # if tagFrame is starting this frame...
        if tagFrame.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            tagFrame.frameNStart = frameN  # exact frame index
            tagFrame.tStart = t  # local t and not account for scr refresh
            tagFrame.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tagFrame, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'tagFrame.started')
            # update status
            tagFrame.status = STARTED
            tagFrame.setAutoDraw(True)
        
        # if tagFrame is active this frame...
        if tagFrame.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Poster.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Poster.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Poster" ---
    for thisComponent in Poster.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Poster
    Poster.tStop = globalClock.getTime(format='float')
    Poster.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Poster.stopped', Poster.tStop)
    thisExp.nextEntry()
    # the Routine "Poster" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Finish" ---
    # create an object to store info about Routine Finish
    Finish = data.Routine(
        name='Finish',
        components=[End, distractor_end],
    )
    Finish.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Finish
    Finish.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Finish.tStart = globalClock.getTime(format='float')
    Finish.status = STARTED
    thisExp.addData('Finish.started', Finish.tStart)
    Finish.maxDuration = None
    # keep track of which components have finished
    FinishComponents = Finish.components
    for thisComponent in Finish.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Finish" ---
    Finish.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 12.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *End* updates
        
        # if End is starting this frame...
        if End.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
            # keep track of start time/frame for later
            End.frameNStart = frameN  # exact frame index
            End.tStart = t  # local t and not account for scr refresh
            End.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(End, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'End.started')
            # update status
            End.status = STARTED
            End.setAutoDraw(True)
        
        # if End is active this frame...
        if End.status == STARTED:
            # update params
            pass
        
        # if End is stopping this frame...
        if End.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > End.tStartRefresh + 7-frameTolerance:
                # keep track of stop time/frame for later
                End.tStop = t  # not accounting for scr refresh
                End.tStopRefresh = tThisFlipGlobal  # on global time
                End.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'End.stopped')
                # update status
                End.status = FINISHED
                End.setAutoDraw(False)
        
        # *distractor_end* updates
        
        # if distractor_end is starting this frame...
        if distractor_end.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            distractor_end.frameNStart = frameN  # exact frame index
            distractor_end.tStart = t  # local t and not account for scr refresh
            distractor_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(distractor_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'distractor_end.started')
            # update status
            distractor_end.status = STARTED
            distractor_end.setAutoDraw(True)
        
        # if distractor_end is active this frame...
        if distractor_end.status == STARTED:
            # update params
            pass
        
        # if distractor_end is stopping this frame...
        if distractor_end.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > distractor_end.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                distractor_end.tStop = t  # not accounting for scr refresh
                distractor_end.tStopRefresh = tThisFlipGlobal  # on global time
                distractor_end.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'distractor_end.stopped')
                # update status
                distractor_end.status = FINISHED
                distractor_end.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Finish.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Finish.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Finish" ---
    for thisComponent in Finish.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Finish
    Finish.tStop = globalClock.getTime(format='float')
    Finish.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Finish.stopped', Finish.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Finish.maxDurationReached:
        routineTimer.addTime(-Finish.maxDuration)
    elif Finish.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-12.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
