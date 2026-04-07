import torch
import typing
import functools
import hashlib
import json
import os
from piat import meta_resolvepath, meta_telemetry
from piat.core import _Dataset, queryfiledl, blobfiledl
from piat.core import __version__, intDebug, intStrict



class Dataloader(torch.utils.data.DataLoader):
    """
    * The central data loader class, aside from the information below it is best to learn more about it by looking at :ref:`examples <examples_dataloader>`.
    * The most interesting aspects of PIAT data loaders is how they load/process samples as defined by their ``funcStages`` functions.
    * While there are many built-in :ref:`functions <api_stages>` to process samples, you will likely find yourself writing your own eventually.

    Parameters
    ----------
    intBatchsize: typing.Optional[int]
        The number of samples per batch, can be ``None`` to disable batching, is 16 by default.
    intWorkers: int
        How many background processes are used for data loading, is 2 by default.
    intThreads: int
        How many threads each process uses, primarily used to hide S3 latency through parallelism, is 4 by default.
    strQueryfile: typing.Union[dict, typing.List[str]]
        The list of queryfiles, either as a list of strings or a dictionary, please see the notes below.
    strBlobfile: typing.List[str]
        The list of blobfiles that should be downloaded to ``strTempdir`` prior to using them, is ``[]`` by default.
    strTempdir: str
        Where to download the queryfiles to prior to using them, is ``'/mnt/localssd/'`` by default with a fallback to ``tempfile.gettempdir()``.
    intSeed: int
        The seed from which all randomness is derived, is ``0`` by default but please see the notes on randomness below.
    boolDroplast: bool
        Whether or not to drop any last batches that are not full when using ``.random()`` or ``.sequential()``, is ``False`` by default.
    boolPinmem: bool
        Whether or not the data loader will copy the tensors into `pinned <https://pytorch.org/docs/stable/data.html#memory-pinning>`__ memory, is ``True`` by default.
    intWorldrank: typing.Optional[int]
        Can be used to manually specify the world rank which is rarely necessary, is ``torch.distributed.get_rank()`` by default.
    intWorldsize: typing.Optional[int]
        Can be used to manually specify the world size which is rarely necessary, is ``torch.distributed.get_world_size()`` by default.
    funcGroup: typing.Optional[typing.Union[str, typing.Callable]]
        An optional function that assigns samples to groups for batching, see the notes below, is ``None`` by default.
    funcWeight: typing.Optional[typing.Union[str, typing.Callable]]
        An optional function that weights samples to make them more/less important, see the notes, is ``None`` by default.
    funcStages: typing.List[typing.Union[str, typing.Callable, dict]]
        The list of functions that load/process a sample, see :ref:`api_stages` for more details.
    funcIntercept: typing.Optional[typing.Union[str, typing.Callable]]
        An optional function that intercepts batches before they are returned, see the notes, is ``None`` by default.

    Notes
    -----
    * This data loader is endless, it will forever loop and never finish, which is also why it does not have a ``len(objDataloader)``.
    * While it internally keeps track of epochs to ensure that each epoch has a new random order, it does not expose this concept.
    * This is due to samples possibly disappearing at any time due to GDPR deletions at which point subsequent epochs would be shorter.
    * If you want to use a debugger, try setting ``intWorkers`` and ``intThreads`` to zero which will make it run in the caller process.

    Notes
    -----
    * If the provided ``strQueryfiles`` is a dictionary, then the keys define the queryfiles and the values the corresponding weight.
    * When using weighted queryfiles, the weights must be a positive integers, please see the provided example below.
    * Underneath the hood, weighted queryfiles effectively just get replicated N times according to their weight (in an efficient way).
    * When using weights, make sure that the total length of the them after weighting is less then 10 billion to not overwhelm the sampler.

    Notes
    -----
    * While the data loader is deterministically random with respect to ``intSeed``, samples get returned as soon as they are ready.
    * This means that the data loader is not perfectly deterministic since samples may appear a batch sooner/later than the last time.
    * You can set ``intThreads == 1`` to be deterministic but this will also be much slower, and is also not guaranteed deterministic.
    * Even with ``intThreads == 1`` a sample may become unavailable (for example due to GDPR deletions) and will not be returned anymore.

    Notes
    -----
    * Sometimes it is important that all samples in a given batch share a certain property, this is what ``funcGroup`` is for.
    * Like aspect ratio bucketing, where we want all samples in a batch to have a similar aspect ratio (to reduce the need for cropping).
    * Since aspect ratio bucketing is a common technique, there is a built-in ``piat.meta_groupaspects`` as shown in the examples below.
    * To make your own grouping, simply write a function that accepts ``objSample`` and returns the group number (integer) for this sample.
    * The data loader also needs to know how many groups there are, so if ``objSample`` is ``None`` then return the number of groups.
    * Lastly, if a ``funcGroup`` does not know how to assign a sample to a group yet then it can return ``'not yet assigned'``.
    * When returning ``'not yet assigned'`` then the grouping function will be called after each ``funcStage`` until it is assigned.

    Notes
    -----
    * Sometimes there are samples that are more/less important than others and ``funcWeight`` can be utilized to weight them differently.
    * To make such a weighting, simply write a function that accepts ``objSample`` and returns the weight (integer) for this sample.
    * The data loader also needs to know what the maximum weight is, so if ``objSample`` is ``None`` then return the maximum weight.
    * You can find an example below, where we have a ``funcWeight`` that weights each sample based on which queryfile it originates from.
    * Lastly, note that the weights should ideally be small (definitely less than 100) to avoid possible data loader slowdowns.

    Notes
    * When executing the ``funcStages`` for a sample, a user might want specific stages to be used only for some samples but not all.
    * To do this, simply provide a dictionary with the condition in form of a string as the key as the conditonal function as the value.
    * Please see the "mix a piat.Dataset with an official dataset/queryfile" example data loader to see a concrete example.

    Notes
    -----
    * Sometimes samples are not ought to be returned as they are, for example, because we want to pack multiple samples together.
    * Token packing is a common `technique <https://pytorch.org/torchtune/0.2/generated/torchtune.datasets.PackedDataset.html>`__ where we do such a repacking to improve downstream efficiency.
    * Please see the :ref:`examples_dataloader` section to find an example that uses ``funcInterept`` to perform this token packing.
    * It boils down the samples from the data loader being given to a ``funcIntercept`` in form of an iterator before they are being collated.

    Examples
    --------
    >>> objDataloader = piat.Dataloader(
    >>>     intBatchsize=16,
    >>>     intWorkers=2,
    >>>     intThreads=4,
    >>>     strQueryfile=[
    >>>         's3://sniklaus-clio-query/*/origin=me',
    >>>         's3://sniklaus-clio-query/*/origin=iw',
    >>>         's3://sniklaus-clio-query/*/origin=fl',
    >>>     ],
    >>>     funcStages=[
    >>>         functools.partial(piat.image_load, {'strSource': '256-pil-antialias'}),
    >>>         functools.partial(piat.image_alpha_smartskip, {}),
    >>>         functools.partial(piat.image_resize_antialias, {'intSize': 128}),
    >>>         functools.partial(piat.image_crop_smart, {'intSize': 128}),
    >>>         functools.partial(piat.text_load, {}),
    >>>         functools.partial(piat.output_image, {'fltMean': [0.5, 0.5, 0.5], 'fltStd': [0.5, 0.5, 0.5]}),
    >>>         functools.partial(piat.output_text, {}),
    >>>     ]
    >>> )
    >>> 
    >>> for objBatch in objDataloader:
    >>>     print(objBatch['tenImage'].shape, objBatch['strText'])

    >>> objDataloader = piat.Dataloader(
    >>>     intBatchsize=16,
    >>>     intWorkers=2,
    >>>     intThreads=4,
    >>>     strQueryfile={
    >>>         's3://sniklaus-clio-query/*/origin=me': 1,
    >>>         's3://sniklaus-clio-query/*/origin=iw': 2,
    >>>         's3://sniklaus-clio-query/*/origin=fl': 3,
    >>>     },
    >>>     funcGroup=functools.partial(piat.meta_groupaspects, {'intSize': 128, 'intMultiple': 16, 'strResize': 'preserve-area'}),
    >>>     funcStages=[
    >>>         functools.partial(piat.image_load, {'strSource': '256-pil-antialias'}),
    >>>         functools.partial(piat.image_alpha_smartskip, {}),
    >>>         functools.partial(piat.image_resize_antialias, {'intSize': 128}),
    >>>         functools.partial(piat.image_crop_smart, {'intSize': 128}),
    >>>         functools.partial(piat.text_load, {}),
    >>>         functools.partial(piat.output_image, {'fltMean': [0.5, 0.5, 0.5], 'fltStd': [0.5, 0.5, 0.5]}),
    >>>         functools.partial(piat.output_text, {}),
    >>>     ]
    >>> )
    >>> 
    >>> for objBatch in objDataloader:
    >>>     print(objBatch['tenImage'].shape, objBatch['strText'])

    >>> def weight(objSample):
    >>>     if objSample is None:
    >>>         return 3
    >>> 
    >>>     if 'origin=me' in objSample['strQueryfile']: return 1
    >>>     if 'origin=iw' in objSample['strQueryfile']: return 2
    >>>     if 'origin=fl' in objSample['strQueryfile']: return 3
    >>> 
    >>> objDataloader = piat.Dataloader(
    >>>     intBatchsize=16,
    >>>     intWorkers=2,
    >>>     intThreads=4,
    >>>     strQueryfile=[
    >>>         's3://sniklaus-clio-query/*/origin=me',
    >>>         's3://sniklaus-clio-query/*/origin=iw',
    >>>         's3://sniklaus-clio-query/*/origin=fl',
    >>>     ],
    >>>     funcWeight=weight,
    >>>     funcStages=[
    >>>         functools.partial(piat.image_load, {'strSource': '256-pil-antialias'}),
    >>>         functools.partial(piat.image_alpha_smartskip, {}),
    >>>         functools.partial(piat.image_resize_antialias, {'intSize': 128}),
    >>>         functools.partial(piat.image_crop_smart, {'intSize': 128}),
    >>>         functools.partial(piat.text_load, {}),
    >>>         functools.partial(piat.output_image, {'fltMean': [0.5, 0.5, 0.5], 'fltStd': [0.5, 0.5, 0.5]}),
    >>>         functools.partial(piat.output_text, {}),
    >>>     ]
    >>> )
    >>> 
    >>> for objBatch in objDataloader:
    >>>     print(objBatch['tenImage'].shape, objBatch['strText'])
    """

    def __init__(self,
        intBatchsize: typing.Optional[int] = 16,
        intWorkers: int = 2,
        intThreads: int = 4,
        strQueryfile: typing.Union[dict, typing.List[str]] = None,
        strBlobfile: typing.List[str] = [],
        strTempdir: str = '/mnt/localssd/',
        intSeed: int = 0,
        boolDroplast: bool = False,
        boolPinmem: bool = True,
        intWorldrank: typing.Optional[int] = None,
        intWorldsize: typing.Optional[int] = None,
        funcGroup: typing.Optional[typing.Union[str, typing.Callable]] = None,
        funcWeight: typing.Optional[typing.Union[str, typing.Callable]] = None,
        funcStages: typing.List[typing.Union[str, typing.Callable, dict]] = None,
        funcIntercept: typing.Optional[typing.Union[str, typing.Callable]] = None,
        collate_fn: typing.Optional[typing.Callable] = None,
    ):
        objArgs = locals()

        for strArg in list(objArgs):
            if 'omegaconf' in str(type(objArgs[strArg])).lower():
                objArgs[strArg] = __import__('omegaconf').OmegaConf.to_object(objArgs[strArg]) # the AdobeDiffusion repo incorrectly passes omegaconf stuff
            # end
        # end

        self.intBatchsize = objArgs['intBatchsize']
        self.intWorkers = objArgs['intWorkers']
        self.intThreads = objArgs['intThreads']
        self.strQueryfile = objArgs['strQueryfile']
        self.strBlobfile = objArgs['strBlobfile']
        self.strTempdir = objArgs['strTempdir']
        self.intSeed = objArgs['intSeed']
        self.boolDroplast = objArgs['boolDroplast']
        self.boolPinmem = objArgs['boolPinmem']
        self.intWorldrank = objArgs['intWorldrank'] if objArgs['intWorldrank'] is not None else torch.distributed.get_rank() if torch.distributed.is_initialized() == True else 0
        self.intWorldsize = objArgs['intWorldsize'] if objArgs['intWorldsize'] is not None else torch.distributed.get_world_size() if torch.distributed.is_initialized() == True else 1
        self.funcGroup = objArgs['funcGroup']
        self.funcWeight = objArgs['funcWeight']
        self.funcStages = objArgs['funcStages']
        self.funcIntercept = objArgs['funcIntercept']

        self.strQueryfile = self.strQueryfile if type(self.strQueryfile) == dict else {strQueryfile: None for strQueryfile in self.strQueryfile} if type(self.strQueryfile) == list else {self.strQueryfile: None}
        self.strQueryfile = {(eval(strQueryfile.replace('piat.', '')) if type(strQueryfile) == str and 'piat.Dataset' in strQueryfile else strQueryfile): intWeight for strQueryfile, intWeight in self.strQueryfile.items()}
        self.strQueryfile = {meta_resolvepath(strQueryfile, ['*-lock']) if type(strQueryfile) == str else strQueryfile: intWeight for strQueryfile, intWeight in self.strQueryfile.items()}

        self.strBlobfile = meta_resolvepath(self.strBlobfile, ['*-lock', '*-ofst', '*-data']) if type(self.strBlobfile) == str else [meta_resolvepath(strBlobfile, ['*-lock', '*-ofst', '*-data']) for strBlobfile in self.strBlobfile]

        with torch.multiprocessing.Pool(8) as objMultipool:
            for objDownload in objMultipool.starmap(queryfiledl, [(strQueryfile, self.strTempdir) for strQueryfile in ([self.strQueryfile] if type(self.strQueryfile) == str else self.strQueryfile)]):
                pass
            # end

            for objDownload in objMultipool.starmap(blobfiledl, [(strBlobfile, self.strTempdir) for strBlobfile in ([self.strBlobfile] if type(self.strBlobfile) == str else self.strBlobfile)]):
                pass
            # end
        # end

        self.collate_fn = collate_fn

        self.objDataset = _Dataset(
            intBatchsize=self.intBatchsize,
            intWorkers=self.intWorkers,
            intThreads=self.intThreads,
            strQueryfile=self.strQueryfile,
            strTempdir=self.strTempdir,
            intSeed=self.intSeed,
            intWorldrank=self.intWorldrank,
            intWorldsize=self.intWorldsize,
            funcGroup=self.funcGroup,
            funcWeight=self.funcWeight,
            funcStages=self.funcStages,
            funcIntercept=self.funcIntercept,
        )

        super().__init__(
            batch_size=self.intBatchsize,
            num_workers=self.intWorkers,
            drop_last=self.boolDroplast if self.intBatchsize is not None else False,
            pin_memory=self.boolPinmem,
            persistent_workers=True if self.intWorkers > 0 else False,
            dataset=self.objDataset,
            collate_fn=self.collate_fn,
        )

        meta_telemetry({
            'strType': 'dataloader',
            'strVersion': __version__,
            'strWorkdir': os.getcwd(),
            'objDataloader': json.loads(repr(self)),
        })

        if intDebug >= 1:
            print(repr(self))
        # end
    # end

    def __repr__(self) -> str:
        with open(os.path.abspath(__file__), 'r') as objFile:
            return json.dumps(obj={
                'strFilehash': hashlib.md5(objFile.read().encode('utf-8')).hexdigest(),
                'intBatchsize': self.intBatchsize,
                'intWorkers': self.intWorkers,
                'intThreads': self.intThreads,
                'strQueryfile': self.strQueryfile,
                'strBlobfile': self.strBlobfile,
                'strTempdir': self.strTempdir,
                'intSeed': self.intSeed,
                'boolDroplast': self.boolDroplast,
                'boolPinmem': self.boolPinmem,
                'intWorldrank': self.intWorldrank,
                'intWorldsize': self.intWorldsize,
                'funcGroup': repr(self.funcGroup),
                'funcWeight': repr(self.funcWeight),
                'funcStages': repr(self.funcStages),
                'funcIntercept': repr(self.funcIntercept),
                'objDataset': json.loads(repr(self.objDataset)) if self.objDataset is not None else None,
            }, indent=2)
        # end
    # end

    def state_dict(self) -> dict:
        """
        * Returns the state of the data loader. We can use ``.load_state_dict(...)`` to recover this state.

        Returns
        -------
        dict

        Notes
        -----
        * During training, it is crucial to not only checkpoint the model weights but also the data loader state.
        * If one were to resume training without recovering the data loader state, one would get the wrong distribution.
        * It would also not be correct to change the data loader seed instead of recovering the data loader state.
        * If there are multiple data loaders, for example when training on multiple GPUs, then checkpoint each one.
        * When resuming training each data loader then also needs to recover its corresponding state.
        * To simplify handling multiple data loaders, consider taking a look at ``piat.meta_mergestates``.
        """

        return self.objDataset.state_dict()
    # end

    def load_state_dict(self, objState: dict) -> object:
        """
        * Sets the data loader to the provided state, see ``.state_dict()`` for more details.

        Parameters
        ----------
        objState: dict
            The data loader state that is ought to be resumed from.

        Returns
        -------
        object

        Notes
        -----
        * See the notes for ``.state_dict()`` for general information on data loader checkpointing.
        * Returns a reference to itself (the data loader) to facilitate API chaining.
        """

        self.objDataset.load_state_dict(objState); return self
    # end

    def reset(self):
        """
        * Resets the data loader to the beginning, only useful if you want to restart with the same behavior.
        """

        self.objDataset.reset()
    # end

    def length(self) -> int:
        """
        * Returns the length of the underlying queryfiles that the data loader is ought to load.

        Returns
        -------
        int

        Notes
        -----
        * This is notably similar to but different from a batch size since samples may be filtered out at whim.
        * A given ``funcStages`` config may filter out half the samples, thus making this function purely informational.
        * Underneath the hood it is basically just returning ``len(piat.Queryfiles(self.strQueryfiles))``.
        """

        return self.objDataset.length()
    # end

    def one(self, objSample: typing.Union[int, float], objAux: typing.Optional[dict] = None, boolRetry: bool = False, boolCache: bool = False) -> typing.Optional[dict]:
        """
        * Processes and returns one sample by reading it from the queryfiles and running it through the ``funcStages``.
        * This is useful when wanting to visualize a set of samples, for example to put them in Wandb.

        Parameters
        ----------
        objSample: typing.Union[int, float]
            The index of the sample, we use ``int(round(objSample * self.length()))`` for float indices.
        objAux: typing.Optional[dict]
            Optional data that is added to ``objScratch`` when calling ``funcStages``, is ``None`` by default.
        boolRetry: bool
            Whether to retry loading the sample until it succeeds, see the notes, is ``False`` by default.
        boolCache: bool
            Whether to cache the results and subsequently return the cached versions, is ``False`` by default.

        Returns
        -------
        typing.Optional[dict]

        Notes
        -----
        * When ``boolRetry`` is ``True`` then it will increment ``objSample`` each time after ten failures.
        * This function runs in the caller thread and can hence be subject to a high latency, use caching if possible.
        """

        return self.objDataset.one(objSample, objAux, boolRetry, boolCache)
    # end

    def random(self) -> typing.Iterable[dict]:
        """
        * Processes and returns all samples in the queryfile in a random order.
        * This is useful when wanting to process all samples at most once, for example for data annotation tasks.

        Returns
        -------
        typing.Iterable[dict]

        Notes
        -----
        * This may not return as many samples as there are in the queryfiles due to filtering and errors.
        * But this entirely depends on the ``funcStages``, if there is no filtering and no failures it should return all.
        * You probably want to have ``boolDroplast`` be ``False`` when using this function to avoid missing samples.
        """

        assert self.objDataset.state_dict()['intSamplein'] == -1

        if self.boolDroplast == True:
            strWarning = 'we will not end up returning all samples because boolDroplast is set to True which means that we will drop the last few samples that are not enough to fill a batch'
            print('warning', strWarning)
            assert intStrict == 0, strWarning
        # end

        self.objDataset.objMiscrandom.value = True

        try:
            for objSample in self:
                yield objSample # if we were to to this inside of the Dataset class then multiprocessing would not work
            # end
        except GeneratorExit:
            pass
        # end

        self.objDataset.objMiscrandom.value = False
    # end

    def sequential(self) -> typing.Iterable[dict]:
        """
        * Processes and returns all samples in the queryfile in sequential order.
        * This is useful when wanting to process all samples at most once, for example for data annotation tasks.

        Returns
        -------
        typing.Iterable[dict]

        Notes
        -----
        * This may not return as many samples as there are in the queryfiles due to filtering and errors.
        * But this entirely depends on the ``funcStages``, if there is no filtering and no failures it should return all.
        * You probably want to have ``boolDroplast`` be ``False`` when using this function to avoid missing samples.
        """

        assert self.objDataset.state_dict()['intSamplein'] == -1

        if self.boolDroplast == True:
            strWarning = 'we will not end up returning all samples because boolDroplast is set to True which means that we will drop the last few samples that are not enough to fill a batch'
            print('warning', strWarning)
            assert intStrict == 0, strWarning
        # end

        self.objDataset.objMiscsequential.value = True

        try:
            for objSample in self:
                yield objSample # if we were to to this inside of the Dataset class then multiprocessing would not work
            # end
        except GeneratorExit:
            pass
        # end

        self.objDataset.objMiscsequential.value = False
    # end
# end