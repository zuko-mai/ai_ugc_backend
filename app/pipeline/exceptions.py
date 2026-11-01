from app.core.exceptions import AppError


class PipelineError(AppError):
    pass


class PipelineNotFoundError(PipelineError):
    pass


class PipelineOrderError(PipelineError):
    pass


class AssetNotFoundError(PipelineError):
    pass


class AssetUploadError(PipelineError):
    pass
